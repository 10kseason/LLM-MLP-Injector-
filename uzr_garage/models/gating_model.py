import torch
import torch.nn as nn

class UZRGate(nn.Module):
    """
    Gating Network for UZR.
    Decides how much to trust the UZR Controller based on Student's context and UZR's own confidence.
    """
    def __init__(self, hidden_dim=896, extra_dim=2, inner_dim=256):
        """
        Args:
            hidden_dim: Dimension of Student's hidden state (e.g., 896 for Qwen-0.5B)
            extra_dim: Dimension of extra features (e.g., entropy, max_attn)
            inner_dim: Hidden dimension of the MLP
        """
        super().__init__()
        self.input_dim = hidden_dim + extra_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, inner_dim),
            nn.ReLU(),
            nn.LayerNorm(inner_dim),
            nn.Dropout(0.1),
            nn.Linear(inner_dim, 1),
            nn.Sigmoid() # Output gate value in [0, 1]
        )
        
    def forward(self, hidden_state, extra_features):
        """
        Args:
            hidden_state: (B, H) Pooled hidden state from Student
            extra_features: (B, E) Extra scalar features (entropy, etc.)
            
        Returns:
            gate: (B, 1) Gating value
        """
        # Concatenate hidden state and extra features
        x = torch.cat([hidden_state, extra_features], dim=-1)
        gate = self.net(x)
        return gate
