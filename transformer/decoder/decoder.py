import torch
import torch.nn as nn
from .decoder_layer import DecoderLayer
from ..positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.p_e = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_out, trgt_padding_mask:None, enc_padding_mask:None):
        x = self.embedding(x)
        x = self.p_e(x)
        for layer in self.layers:
            x = layer(x, enc_out, trgt_padding_mask, enc_padding_mask)
        x = self.linear(x)
        return x
