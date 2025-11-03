import torch
import torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(self.max_len, d_model) # shape: [max_len, d_model]
        position = torch.arange(0, self.max_len).reshape(-1,1)  # shape: [max_len, 1]
        div_term = torch.pow(10000, torch.arange(0, self.d_model, 2)/self.d_model) #shape: [1, d_model/2]

        pe[:, 0::2] = torch.sin(position/div_term)
        pe[:, 1::2] = torch.cos(position/div_term[:d_model//2])

        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]].to(x.device)
        return x