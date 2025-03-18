import torch.nn as nn
import torch.nn.functional as F

class Regressor(nn.Module):
    def __init__(self, in_dim, out_dim, filter=[100, 300, 100]):
        super(Regressor, self).__init__()
        self.filter = filter
        self.fc1 = nn.Linear(in_dim, self.filter[0], bias=True)
        self.bn1 = nn.BatchNorm1d(self.filter[0])
        self.fc2 = nn.Linear(self.filter[0], self.filter[1], bias=True)
        self.bn2 = nn.BatchNorm1d(self.filter[1])
        self.fc3 = nn.Linear(self.filter[1], self.filter[2], bias=True)
        self.bn3 = nn.BatchNorm1d(self.filter[2])
        self.fc4 = nn.Linear(self.filter[2], out_dim, bias=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x