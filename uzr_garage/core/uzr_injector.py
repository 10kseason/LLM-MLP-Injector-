import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
import types

class UZRInjector:
    @staticmethod
    def inject(student_model, controller_model, target_layer_idx=12, alpha=1.0):
        """
        student_model: Qwen2.5 0.5B
        controller_model: UZRController (학습된 가중치 로드된 상태)
        target_layer_idx: 주입할 레이어 인덱스
        alpha: 전역 최대 게인 (0~1 권장, 실험용으로 0.3~0.7 추천)
        """
        target_layer = student_model.model.layers[target_layer_idx]
        attn_mod = target_layer.self_attn

        attn_mod.uzr_controller = controller_model
        attn_mod.uzr_alpha_max = alpha
        attn_mod.uzr_enabled = True
        attn_mod.uzr_target_layer_idx = target_layer_idx
        attn_mod.uzr_last_stats = {} # Initialize stats

        # forward 갈아끼우기
        attn_mod.forward = types.MethodType(uzr_forward_prob_mix, attn_mod)

        print(f"[UZR] Injected at layer {target_layer_idx} | alpha_max={alpha}")

    @staticmethod
    def eject(student_model, target_layer_idx):
        target_layer = student_model.model.layers[target_layer_idx]
        attn_mod = target_layer.self_attn
        
        if hasattr(attn_mod, "_original_forward"):
            attn_mod.forward = attn_mod._original_forward
            attn_mod.uzr_enabled = False
            print(f"[UZR] Ejected from Layer {target_layer_idx}.")

def uzr_forward_prob_mix(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor = None,
    position_ids: torch.LongTensor = None,
    past_key_value = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """
    Qwen2Attention 커스텀 버전
    - 원래 Attention logit으로 P_base 계산
    - Controller로 P_uzr 계산
    - Entropy 기반 local alpha로 확률 레벨에서 섞기
    """
    # Handle argument naming mismatch (past_key_values vs past_key_value)
    past_key_values = past_key_value
    if past_key_values is None:
        past_key_values = kwargs.get("past_key_values", None)

    bsz, q_len, _ = hidden_states.size()

    # 1. Q, K, V
    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # 2. RoPE
    kv_seq_len = key_states.shape[-2]
    if past_key_values is not None:
        kv_seq_len += past_key_values.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # 3. KV 캐시
    if past_key_values is not None:
        past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs=kwargs
        )
        # Retrieve full sequence from cache
        full_keys, full_values = past_key_values[self.layer_idx]
        key_states = full_keys
        value_states = full_values

    # GQA 확장
    key_states = torch.repeat_interleave(
        key_states, dim=1, repeats=self.num_heads // self.num_key_value_heads
    )
    value_states = torch.repeat_interleave(
        value_states, dim=1, repeats=self.num_heads // self.num_key_value_heads
    )

    # 4. 원래 Attention Logits
    attn_scores = torch.matmul(query_states, key_states.transpose(2, 3))
    attn_scores = attn_scores / math.sqrt(self.head_dim)  # (B, H, Q, K)

    # -------------------- UZR 블록 시작 --------------------
    use_uzr = getattr(self, "uzr_enabled", False) and hasattr(self, "uzr_controller")

    if use_uzr:
        controller = self.uzr_controller
        alpha_max  = getattr(self, "uzr_alpha_max", 0.5)

        # Controller: hidden_states -> logits (B, Q, Max_S)
        # 여기선 kv_seq_len과 q_len이 같다고 가정 (causal LM)
        # [Fix] Cast to float for controller
        uzr_logits = controller(hidden_states.float())            # (B, S, Max_S)
        
        # Slicing to match current KV length
        curr_kv_len = key_states.shape[-2]
        
        # Handle shape mismatch if controller output is smaller/larger
        if uzr_logits.shape[-1] < curr_kv_len:
             # Pad if too short
             pad_len = curr_kv_len - uzr_logits.shape[-1]
             padding = torch.zeros((uzr_logits.shape[0], uzr_logits.shape[1], pad_len), device=uzr_logits.device, dtype=uzr_logits.dtype)
             uzr_logits = torch.cat([uzr_logits, padding], dim=-1)
        
        uzr_logits = uzr_logits[:, :q_len, :curr_kv_len]           # (B, Q, K)

        # Causal mask 적용 (미래는 보지 못하게)
        # Note: During generation (q_len=1), we don't need causal mask for the query itself against past keys, 
        # but the controller output might need masking if it was trained with full context.
        # However, controller outputs a fixed size map.
        # Let's apply a simple causal mask if q_len > 1 (prefill).
        if q_len > 1:
            causal = torch.tril(torch.ones(q_len, curr_kv_len, device=hidden_states.device))
            uzr_logits = uzr_logits.masked_fill(causal.unsqueeze(0) == 0, -1e9)

        # Base logits에도 mask 적용해서 P_base 만들 준비
        base_logits = attn_scores
        if attention_mask is not None:
            base_logits = base_logits + attention_mask  # (B, H, Q, K)

        # Softmax → 확률 분포로
        base_probs = F.softmax(base_logits, dim=-1, dtype=torch.float32)  # (B, H, Q, K)

        uzr_probs  = F.softmax(uzr_logits, dim=-1, dtype=torch.float32)   # (B, Q, K)

        # Uzr 분포 기반 엔트로피 계산 (Hypothesis 1 + 2)
        # H = -sum p log p  ∈ [0, log(K)]
        entropy = -(uzr_probs * uzr_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)  # (B, Q, 1)
        max_entropy = math.log(curr_kv_len + 1e-8)
        norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0)  # 0(샤프) ~ 1(완전 플랫)

        # 샤프할수록 alpha 커지게: alpha_local = alpha_max * (1 - H_norm)
        alpha_local = alpha_max * (1.0 - norm_entropy)          # (B, Q, 1)
        
        # [Stats] Store gating statistics for analysis
        # We store the mean alpha across the current query sequence
        if not hasattr(self, "uzr_history"):
            self.uzr_history = []
        
        # Detach to avoid memory leak if we store history
        curr_alpha_mean = alpha_local.mean().item()
        self.uzr_last_stats = {
            "alpha_mean": curr_alpha_mean,
            "entropy_mean": entropy.mean().item()
        }
        
        alpha_local = alpha_local.unsqueeze(1)                   # (B, 1, Q, 1)

        # uzr_probs를 head 차원으로 broadcast
        uzr_probs_h = uzr_probs.unsqueeze(1)                     # (B, 1, Q, K)

        # 최종 섞기: P_mix = (1-a) * P_base + a * P_uzr
        mixed_probs = (1.0 - alpha_local) * base_probs + alpha_local * uzr_probs_h

        # 수치 안정화 위해 다시 정규화
        mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        attn_probs = mixed_probs.to(query_states.dtype)  # (B, H, Q, K)

    else:
        # 원래 경로 (UZR 미사용)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # -------------------- UZR 블록 끝 --------------------

    # 5. V와 곱해서 출력
    attn_output = torch.matmul(attn_probs, value_states)  # (B, H, Q, d_head)
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_probs_out = None
    else:
        attn_probs_out = attn_probs

    return attn_output, attn_probs_out, past_key_values
