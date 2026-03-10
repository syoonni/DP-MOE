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
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        self.linear_layers = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(2)
        ])
        
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, expert_outputs):
        batch_size, num_experts, feature_dim = expert_outputs.shape

        attn_output, _ = self.attention(
            expert_outputs, expert_outputs, expert_outputs
        )

        attn_output = self.norm(expert_outputs + attn_output)

        final_outputs = []
        for i in range(num_experts):
            linear_out = self.linear_layers[i](attn_output[:, i])
            expert_out = linear_out + attn_output[:, i]
            final_outputs.append(expert_out)
        return torch.stack(final_outputs, dim=1)


class ResidualMOE(nn.Module):
    def __init__(self, expert_dims, out_dim):
        super(ResidualMOE, self).__init__()

        self.experts = nn.ModuleDict()
        self.expert_names = list(expert_dims.keys())
        self.num_experts = len(expert_dims)

        for name, dim in expert_dims.items():
            self.experts[name] = Expert(dim, out_dim)

        hidden_dim = 100

        self.attention_gate = SelfAttentionGate(
            feature_dim=hidden_dim, 
            num_heads=2
        )

        self.combiner = nn.Linear(hidden_dim * self.num_experts, out_dim)

        total_dim = sum(expert_dims.values())
        self.residual_processor = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, inputs):
        """
        Forward pass with self-attention in gate mechanism
        Args:
            inputs: Dictionary with expert names and their input tensors
        """
        expert_outputs = {}
        expert_hidden_states = {}

        for name in self.expert_names:
            logits, hidden = self.experts[name](inputs[name])
            expert_outputs[name] = F.softmax(logits, dim=1)
            expert_hidden_states[name] = hidden

        stacked_hidden = torch.stack([expert_hidden_states[name] for name in self.expert_names], dim=1)

        attended_hidden = self.attention_gate(stacked_hidden)

        processed_hidden = {}
        for i, name in enumerate(self.expert_names):
            processed_hidden[name] = attended_hidden[:, i]

        combined_hidden = torch.cat([processed_hidden[name] for name in self.expert_names], dim=1)
        combined_out = self.combiner(combined_hidden)
        combined_out = F.softmax(combined_out, dim=1)

        input_features = torch.cat([inputs[name] for name in self.expert_names], dim=1)
        residual_out = self.residual_processor(input_features)
        residual_out = F.softmax(residual_out, dim=1)

        final_output = self.alpha * combined_out + (1 - self.alpha) * residual_out

        gate_weights = torch.zeros(input_features.size(0), self.num_experts, device=input_features.device)
        for i, name in enumerate(self.expert_names):
            gate_weights[:, i] = processed_hidden[name].norm(dim=1) / combined_hidden.norm(dim=1)
        gate_weights = F.softmax(gate_weights, dim=1)
        
        return final_output, gate_weights, expert_outputs, residual_out
