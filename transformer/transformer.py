import torch
import torch.nn as nn
from .encoder.encoder import Encoder
from .decoder.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, tokenizer, d_model, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.encoder = Encoder(self.vocab_size, d_model, num_heads, num_layers, dropout)
        self.decoder = Decoder(self.vocab_size, d_model, num_heads, num_layers, dropout)

    def forward(self, x, y):

        src_padding_mask = (x == 0).to(x.device)
        trgt_padding_mask = (y == 0).to(y.device)

        encoder_output = self.encoder(x, src_padding_mask=src_padding_mask)

        logits = self.decoder(y, encoder_output, trgt_padding_mask=trgt_padding_mask, enc_padding_mask=src_padding_mask)

        return logits