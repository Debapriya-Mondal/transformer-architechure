import torch
import torch.nn as nn
from .encoder_layer import EncoderLayer
from ..positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.p_e = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_padding_mask=None):
        x = self.embedding(x)
        x = self.p_e(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_padding_mask=src_padding_mask)
        return x