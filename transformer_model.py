import torch
import torch.nn as nn

class WordTransformer(nn.Module):
    def __init__(self, d_model=16, nhead=2, num_layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
    def forward(self, src):
        return self.transformer(src.float())