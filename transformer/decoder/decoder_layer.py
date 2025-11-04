import torch
import torch.nn as nn
from ..linear_layer import FFN
from ..attention.self_attention import MultiHeadSelfAttention
from ..attention.cross_attention import MultiHeadCrossAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, masked=True, dropout=dropout)
        self.cross_mhsa = MultiHeadCrossAttention(d_model, num_heads, dropout=dropout)
        self.ffn = FFN(d_model)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, trgt_padding_mask: None, enc_padding_mask:None):
        x = x + self.dropout1(self.mhsa(x, padding_mask=trgt_padding_mask))
        x = self.layer_norm1(x)

        x = x + self.dropout2(self.cross_mhsa(x, enc_out, enc_padding_mask=enc_padding_mask))
        x = self.layer_norm2(x)

        x = x + self.dropout3(self.ffn(x))
        x = self.layer_norm3(x)
        return x