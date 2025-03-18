import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, in_dim, out_dim, filters=[100, 300, 100]):
        super(Expert, self).__init__()
        self.filters = filters
        
        self.fc1 = nn.Linear(in_dim, self.filters[0], bias=True)
        self.bn1 = nn.BatchNorm1d(self.filters[0])
        self.fc2 = nn.Linear(self.filters[0], self.filters[1], bias=True)
        self.bn2 = nn.BatchNorm1d(self.filters[1])
        self.fc3 = nn.Linear(self.filters[1], self.filters[2], bias=True)
        self.bn3 = nn.BatchNorm1d(self.filters[2])
        self.fc4 = nn.Linear(self.filters[2], out_dim, bias=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        hidden = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(hidden)
        logits = self.fc4(x)

        return logits, hidden


class SelfAttentionGate(nn.Module):
    def __init__(self, feature_dim, num_heads=2):
        super(SelfAttentionGate, self).__init__()
        # Self-attention 레이어 (전문가 출력 간의 관계 모델링)
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        # 각 전문가 출력에 대한 개별 처리 레이어
        self.linear_layers = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(2)
        ])
        
        # 정규화 레이어
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, expert_outputs):
        # expert_outputs: [batch_size, num_experts, feature_dim]
        batch_size, num_experts, feature_dim = expert_outputs.shape
        
        # Self-attention 적용
        attn_output, _ = self.attention(
            expert_outputs, expert_outputs, expert_outputs
        )
        
        # Residual 연결 및 정규화
        attn_output = self.norm(expert_outputs + attn_output)
        
        # 각 전문가 출력을 개별적으로 처리
        final_outputs = []
        for i in range(num_experts):
            # Linear 변환
            linear_out = self.linear_layers[i](attn_output[:, i])
            # Residual 연결
            expert_out = linear_out + attn_output[:, i]
            final_outputs.append(expert_out)
            
        # 처리된 출력을 다시 스택
        return torch.stack(final_outputs, dim=1)


class ResidualMOE(nn.Module):
    def __init__(self, expert_dims, out_dim):
        """
        Initialize MoE model with Self-Attention in Gate Network
        Args:
            expert_dims: Dictionary with expert names and their input dimensions
            out_dim: Output dimension (default: 2 for binary classification)
        """
        super(ResidualMOE, self).__init__()

        self.experts = nn.ModuleDict()
        self.expert_names = list(expert_dims.keys())
        self.num_experts = len(expert_dims)

        # 전문가 생성
        for name, dim in expert_dims.items():
            self.experts[name] = Expert(dim, out_dim)

        # 전문가 은닉층 차원 (마지막 레이어의 크기)
        hidden_dim = 100  # Expert의 filters[2] 값
        
        # Self-attention을 포함한 게이트 네트워크
        self.attention_gate = SelfAttentionGate(
            feature_dim=hidden_dim, 
            num_heads=2
        )
        
        # 전문가 출력을 결합하여 최종 예측을 만드는 레이어
        self.combiner = nn.Linear(hidden_dim * self.num_experts, out_dim)
        
        # 입력 특성들을 직접 처리하는 residual 연결
        total_dim = sum(expert_dims.values())
        self.residual_processor = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Residual과 Expert 출력의 비중을 조절하는 파라미터
        self.alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, inputs):
        """
        Forward pass with self-attention in gate mechanism
        Args:
            inputs: Dictionary with expert names and their input tensors
        """
        expert_outputs = {}
        expert_hidden_states = {}
        
        # 각 전문가의 출력 계산
        for name in self.expert_names:
            logits, hidden = self.experts[name](inputs[name])
            expert_outputs[name] = F.softmax(logits, dim=1)
            expert_hidden_states[name] = hidden
        
        # 전문가 hidden states를 스택하여 attention에 전달
        stacked_hidden = torch.stack([expert_hidden_states[name] for name in self.expert_names], dim=1)
        
        # Self-attention 적용
        attended_hidden = self.attention_gate(stacked_hidden)
        
        # Attended hidden states 추출
        processed_hidden = {}
        for i, name in enumerate(self.expert_names):
            processed_hidden[name] = attended_hidden[:, i]
        
        # 처리된 hidden states를 연결하여 최종 예측 생성
        combined_hidden = torch.cat([processed_hidden[name] for name in self.expert_names], dim=1)
        combined_out = self.combiner(combined_hidden)
        combined_out = F.softmax(combined_out, dim=1)
        
        # Residual 연결 계산
        input_features = torch.cat([inputs[name] for name in self.expert_names], dim=1)
        residual_out = self.residual_processor(input_features)
        residual_out = F.softmax(residual_out, dim=1)
        
        # Expert 출력과 Residual 출력 결합
        final_output = self.alpha * combined_out + (1 - self.alpha) * residual_out
        
        # 원래 인터페이스와 호환되도록 전문가 게이트 가중치 계산 (평균 attention 가중치 사용)
        # 여기서는 실제 attention 가중치 대신 processed_hidden 값의 norm을 사용
        gate_weights = torch.zeros(input_features.size(0), self.num_experts, device=input_features.device)
        for i, name in enumerate(self.expert_names):
            gate_weights[:, i] = processed_hidden[name].norm(dim=1) / combined_hidden.norm(dim=1)
        gate_weights = F.softmax(gate_weights, dim=1)
        
        return final_output, gate_weights, expert_outputs, residual_out