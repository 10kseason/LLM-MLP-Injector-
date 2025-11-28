import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class StudentWithController(nn.Module):
    def __init__(self, student_model_name, controller_model, target_layer=8, alpha=0.1, device="cuda"):
        super().__init__()
        self.student = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            device_map=device,
            torch_dtype=torch.float16,
            attn_implementation="eager" # Needed for easy injection/hooking
        )
        self.controller = controller_model
        self.target_layer = target_layer
        self.alpha = alpha
        self.device = device
        
        # We need to capture the input to the target layer to feed the controller
        # And then inject the output of the controller into the attention of that layer.
        
        # Strategy:
        # 1. Hook Pre-Forward of Layer N: Capture hidden state -> Run Controller -> Get Bias
        # 2. Hook Forward of Attention of Layer N: Add Bias to attn_weights
        
        self.controller_bias = None
        # self._register_hooks() # Replaced by _wrap_attention
        self._wrap_attention()

    # def _pre_layer_hook(self, module, args): ... (removed)
    # def _attn_forward_hook(self, module, input, output): ... (removed)
    # def _register_hooks(self): ... (removed)
    # def _modified_attn_forward(self, ...): ... (removed)
    
    # We keep _wrap_attention and forward


    def forward(self, input_ids, attention_mask=None):
        # We need to ensure the hook is active and has access to the controller.
        # The hook uses `self.controller_bias`.
        
        # We need to reset bias
        self.controller_bias = None
        
        return self.student(input_ids, attention_mask=attention_mask)

    # We need to implement the actual modification logic.
    # Since I cannot easily modify the internal `attn_weights` without copy-pasting code,
    # I will try to use the `attention_mask` injection via a wrapper class for the Attention module.
    
    def _wrap_attention(self):
        layer = self.student.model.layers[self.target_layer]
        original_attn = layer.self_attn
        
        class WrappedAttention(nn.Module):
            def __init__(self, original, parent):
                super().__init__()
                self.original = original
                self.parent = parent # StudentWithController instance
                
            def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
                # 1. Run Controller (or get cached bias)
                # We can't easily run controller here because we need input to the LAYER, not the ATTENTION module (which is normalized usually).
                # But hidden_states passed to attention IS the input (after layernorm usually).
                # Let's use this hidden_states to drive the controller.
                
                if self.parent.controller_bias is None:
                    # Compute bias on the fly
                    with torch.no_grad():
                        # hidden_states: (B, S, D)
                        logits = self.parent.controller(hidden_states)
                        self.parent.controller_bias = logits * self.parent.alpha
                
                bias = self.parent.controller_bias # (B, S, S)
                
                # 2. Inject into attention_mask
                # attention_mask is (B, 1, S, S) or similar.
                # We add our bias.
                if attention_mask is None:
                    attention_mask = 0.0
                
                # Ensure shapes match
                # bias: (B, S, S) -> (B, 1, S, S)
                bias_expanded = bias.unsqueeze(1)
                
                # We need to be careful about broadcasting and types.
                # attention_mask might be 4D.
                
                new_mask = attention_mask + bias_expanded
                
                return self.original(hidden_states, attention_mask=new_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, **kwargs)
        
        layer.self_attn = WrappedAttention(original_attn, self)

    # Update __init__ to use _wrap_attention instead of hooks
    # ...

