import torch
import numpy as np

class PatternLogger:
    """
    Extracts and logs patterns from Teacher and Student models for UZR Gating training.
    """
    
    @staticmethod
    def extract_teacher_features(teacher_logits, teacher_generated_ids, tokenizer):
        """
        Extracts features from Teacher's output to determine confidence/caution.
        
        Args:
            teacher_logits: (B, S, V) Logits from teacher
            teacher_generated_ids: (B, S) Token IDs generated
            tokenizer: Tokenizer for decoding
            
        Returns:
            dict: {
                "entropy": float,
                "label": int (0=CONFIDENT, 1=CAUTIOUS, 2=WAFFLE)
            }
        """
        # Calculate Entropy of the generated sequence
        # We focus on the generated part (excluding prompt if logits cover full seq)
        # Assuming logits correspond to the generated tokens
        
        probs = torch.softmax(teacher_logits, dim=-1)
        log_probs = torch.log_softmax(teacher_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item() # Average entropy over sequence
        
        # Simple Heuristic Labeling (v0.1)
        # TODO: Refine this with more sophisticated rules or a trained classifier
        
        text = tokenizer.decode(teacher_generated_ids[0], skip_special_tokens=True).lower()
        length = teacher_generated_ids.shape[1]
        
        # Keywords for caution
        caution_keywords = ["not sure", "don't know", "cannot", "unclear", "depends", "might"]
        is_cautious = any(k in text for k in caution_keywords)
        
        if entropy < 1.5 and not is_cautious and length < 50:
            label = 0 # CONFIDENT
        elif is_cautious or (entropy > 2.0 and length < 80):
            label = 1 # CAUTIOUS
        else:
            label = 2 # WAFFLE (Long, high entropy, or just rambling)
            
        return {
            "entropy": entropy,
            "label": label,
            "length": length
        }

    @staticmethod
    def extract_student_features(student_hidden, uzr_attn_dist):
        """
        Extracts features from Student and UZR Controller.
        
        Args:
            student_hidden: (B, S, H) Hidden state from target layer
            uzr_attn_dist: (B, S, S) Attention distribution from Controller
            
        Returns:
            dict: {
                "hidden_mean": (H,),
                "uzr_entropy": float,
                "uzr_max_attn": float
            }
        """
        # 1. Student Hidden Summary (Mean Pooling)
        # We use the mean of the hidden states as a simple sentence embedding
        hidden_mean = student_hidden.mean(dim=1).squeeze(0).float().cpu().numpy() # (H,)
        
        # 2. UZR Attention Stats
        # Entropy of the attention distribution (per row, then averaged)
        # uzr_attn_dist is (B, S, S)
        # Avoid log(0)
        attn_probs = uzr_attn_dist + 1e-9
        attn_entropy = -(attn_probs * torch.log(attn_probs)).sum(dim=-1).mean().item()
        
        # Max attention weight (peakiness)
        max_attn = attn_probs.max(dim=-1).values.mean().item()
        
        return {
            "hidden_mean": hidden_mean,
            "uzr_entropy": attn_entropy,
            "uzr_max_attn": max_attn
        }
