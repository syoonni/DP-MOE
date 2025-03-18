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
    
    
class MOE(nn.Module):
    def __init__(self, expert_dims, out_dim):
        """
        Initialize MoE model with variable number of experts
        Args:
            expert_dims: Dictionary with expert names and their input dimensions
            out_dim: Output dimension (default: 2 for binary classification)
        """
        super(MOE, self).__init__()

        self.experts = nn.ModuleDict()
        self.expert_names = list(expert_dims.keys())
        self.num_experts = len(expert_dims)

        for name, dim in expert_dims.items():
            self.experts[name] = Expert(dim, out_dim)

        total_dim = sum(expert_dims.values())

        #Improved gate with batch
        self.gate = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, self.num_experts),  # Weight for each expert
            nn.Softmax(dim=1)
        )



    def forward(self, inputs):
        """
        Forward pass
        Args:
            inputs: Dictionary with expert names and their input tensors
        """
        expert_outputs = {}
        expert_features = {}

        for name in self.expert_names:
            logits, hidden = self.experts[name](inputs[name])
            expert_outputs[name] = F.softmax(logits, dim=1)
            expert_features[name] = hidden
        
        gate_input = torch.cat([inputs[name] for name in self.expert_names], dim=1)

        weights = self.gate(gate_input)

        combined_output = torch.zeros_like(expert_outputs[self.expert_names[0]])
        for i, name in enumerate(self.expert_names):
            combined_output += weights[:, i].unsqueeze(1) * expert_outputs[name]

        return combined_output, weights, expert_outputs 