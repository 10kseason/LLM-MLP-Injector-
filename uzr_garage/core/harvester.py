import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

# --- 설정 (Config) ---
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"  # 니가 쓸 똑똑한 선생
STUDENT_ID = "Qwen/Qwen2.5-0.5B"         # 가르칠 멍청한 제자 (0.6B 급)
OUTPUT_DIR = "./data/harvested_logs"

# GPU 메모리 최적화 설정 (4-bit Load)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # 연산은 FP16으로 (속도 UP)
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",             # 노이즈 적은 NF4 포맷
)

class UZRHarvester:
    def __init__(self):
        print(f"[UZR] Loading Teacher: {TEACHER_ID} (4-bit)...")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_ID,
            # quantization_config=bnb_config, # Windows issue
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="eager"  # [중요] Flash Attn 끄고 Raw Map 뽑기
            # [Fix] Eager mode might be unstable in FP16 on Windows. Let Auto choose (likely SDPA).
        )
        self.t_tok = AutoTokenizer.from_pretrained(TEACHER_ID)

        print(f"[UZR] Loading Student: {STUDENT_ID} (FP16)...")
        self.student = AutoModelForCausalLM.from_pretrained(
            STUDENT_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.s_tok = AutoTokenizer.from_pretrained(STUDENT_ID)
        
        # 데이터 임시 저장소
        self.cache = {"student_hidden": {}}
        
        # Student Hook 설치 (예: 중간 레이어 12번)
        # Qwen2.5-0.5B는 총 24 Layer임. 중간인 12번 털어보자.
        target_layer = 12 
        self.student.model.layers[target_layer].register_forward_hook(self.get_student_hook(target_layer))
        print(f"[UZR] Hook registered at Student Layer {target_layer}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def get_student_hook(self, layer_idx):
        def hook(module, input, output):
            # output[0] shape: (Batch, Seq, Hidden)
            self.cache["student_hidden"][layer_idx] = output[0].detach().cpu().half()
        return hook

    def process(self, text, sample_id):
        # 1. Dual Tokenization
        t_inputs = self.t_tok(text, return_tensors="pt").to(self.teacher.device)
        s_inputs = self.s_tok(text, return_tensors="pt").to(self.student.device)

        # 2. Length Check (길이 다르면 나가리)
        t_len = t_inputs.input_ids.shape[1]
        s_len = s_inputs.input_ids.shape[1]
        
        if t_len != s_len:
            # print(f"[Skip] Length Mismatch: T={t_len} vs S={s_len}")
            return False

        # 3. Forward (Inference)
        with torch.no_grad():
            # Teacher: Attention Map 추출
            t_out = self.teacher(**t_inputs, output_attentions=True)
            # t_out.attentions: 튜플 (Layer 수 만큼), 각 요소는 (B, Head, S, S)
            
            # Student: Hook이 알아서 Hidden 채집함
            self.student(**s_inputs)

        # 4. Data Packing (핵심: Teacher 마지막 레이어 Attn만 가져오기 - 용량 절약)
        # (Batch, Head, Seq, Seq) -> Head 평균 -> (Seq, Seq)
        # Teacher의 마지막 레이어(-1)를 타겟으로 함
        # [Fix] FP16 mean operation might be unstable, cast to float32 first
        raw_attn = t_out.attentions[-1][0] # (Head, S, S)
        
        if torch.isnan(raw_attn).any():
            print(f"[Warn] NaN detected in Teacher Attn for sample {sample_id}. Skipping.")
            return False
            
        last_layer_attn = raw_attn.float().mean(dim=0).detach().cpu().half()  
        
        student_hidden = list(self.cache["student_hidden"].values())[0][0] # (Seq, Hidden)
        
        if torch.isnan(student_hidden).any():
            print(f"[Warn] NaN detected in Student Hidden for sample {sample_id}. Skipping.")
            return False

        # 5. Save
        save_path = os.path.join(OUTPUT_DIR, f"sample_{sample_id:04d}.npz")
        np.savez_compressed(
            save_path,
            text=text,
            teacher_attn=last_layer_attn.numpy(),  # (S, S) - 정답지
            student_hidden=student_hidden.numpy()  # (S, H) - 문제지
        )
        print(f"[Saved] {save_path} | Shape: Attn{last_layer_attn.shape}, Hidden{student_hidden.shape}")
        return True

# --- 실행 테스트 ---
# --- 실행 테스트 ---
if __name__ == "__main__":
    from datasets import load_dataset
    from tqdm import tqdm

    harvester = UZRHarvester()
    
    # Load dataset for bulk collection
    print("[UZR] Loading dataset for bulk collection...")
    # Using a small subset of a text dataset. 
    # 'wikitext' or 'code_search_net' or similar. 
    # Let's use 'wikitext-2-raw-v1' for general text or 'code_search_net' for code.
    # User mentioned "Uzr Garage" context, maybe code? Or general text?
    # User said "집에 있는 텍스트 파일 아무거나 읽어서".
    # I'll use wikitext for variety.
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    
    count = 0
    target_count = 100
    
    print(f"[UZR] Collecting {target_count} samples...")
    for item in tqdm(ds):
        text = item["text"]
        if len(text.strip()) < 50: # Skip short/empty
            continue
            
        # Truncate to avoid huge sequences for v0.1
        if len(text) > 500:
            text = text[:500]
            
        success = harvester.process(text, count)
        if success:
            count += 1
            
        if count >= target_count:
            break
            
    print("[UZR] Collection Complete.")
