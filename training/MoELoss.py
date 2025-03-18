import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MoELoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, class_weights=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.alpha = alpha  # Entropy regularization coefficient
        self.beta = beta   # Balance regularization coefficient

    def forward(self, combined_out, expert_outputs, gate_weights, targets, confidences):
        # Main loss
        raw_losses = self.ce_loss(combined_out, targets)
        combined_loss = torch.mean(raw_losses * confidences)
        
        # Gate entropy regularization
        entropy_loss = -torch.mean(torch.sum(gate_weights * torch.log(gate_weights + 1e-10), dim=1))
        
        # Expert balance regularization
        balance_loss = torch.mean(torch.abs(gate_weights.mean(dim=0)))
        
        return combined_loss - self.alpha * entropy_loss + self.beta * balance_loss