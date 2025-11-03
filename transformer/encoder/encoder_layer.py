import torch
import torch.nn as nn
from ..linear_layer import FFN
from ..attention.self_attention import MultiHeadSelfAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FFN(d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_padding_mask=None):
        x = x + self.mhsa(x, padding_mask=src_padding_mask)
        x = self.layer_norm1(x)
        x = x + self.ffn(x)
        x = self.layer_norm2(x)
        return x