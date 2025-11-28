import torch
import torch.nn as nn

class UZRController(nn.Module):
    def __init__(self, input_dim=896, hidden_dim=512, max_seq_len=128):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # 구조: Hidden(896) -> 압축(512) -> 활성 -> 확장(Max_Seq_Len)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max_seq_len) # 출력: 고정된 Max 길이만큼의 Attention Logits
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)
        # output: (Batch, Seq_Len, Max_Seq_Len)
        
        # 1. 예측 (Logits)
        logits = self.net(x)
        
        # 2. 유효한 범위(Seq_Len) 밖은 -infinity 처리해야 하지만,
        # v0.1 학습 단계에서는 Loss 계산 때 마스킹으로 처리함.
        # 추론 때는 softmax 때리기 전에 slicing 필요.
        
        return logits
