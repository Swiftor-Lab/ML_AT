import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    def __init__(self, node_dim=4, hidden_dim=16):
        super().__init__()
        self.linear = nn.Linear(node_dim, hidden_dim)
        
    def forward(self, x, adj):
        h = F.relu(self.linear(x.float()))
        return torch.matmul(adj.float(), h)