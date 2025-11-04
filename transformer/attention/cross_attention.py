import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.d_head = d_model // num_heads   # dimension of each head

        self.Q_W = nn.Linear(d_model, d_model, bias=False) # Query matrix
        self.K_W = nn.Linear(d_model, d_model, bias=False)  # Key matrix
        self.V_W = nn.Linear(d_model, d_model, bias=False)  # Value matrix

        # final Output layer
        self.O_W = nn.Linear(d_model, d_model, bias=False)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, dec_inp, enc_out, enc_padding_mask=None):
        #   batch_size, seq_length, _ = x.shape
        batch_size, dec_len, _ = dec_inp.shape
        _, enc_len, _ = enc_out.shape
        device = dec_inp.device

        # dec_inp: [batch_size, dec_length, d_model]
        # enc_out: [batch_size, enc_length, d_model]

        Q = self.Q_W(dec_inp)  # shape: [batch_size, dec_length, d_model]
        K = self.K_W(enc_out)  # shape: [batch_size, enc_length, d_model]
        V = self.V_W(enc_out) # shape: [batch_size, enc_length, d_model]



        def split_head(x):
            # transpose: batch_size, dec_len or enc_len, num_heads, d_head -> batch_size, num_heads, dec_length or enc_length, d_head
            return x.reshape(batch_size, -1, self.num_heads, self.d_head).transpose(1,2)

        Q_heads = split_head(Q) # shape: [batch_size, num_heads, dec_length, d_head]
        K_heads = split_head(K) # shape: [batch_size, num_heads, enc_length, d_head]
        V_heads = split_head(V) # shape: [batch_size, num_heads, enc_length, d_head]

        # attention calculation
        score = torch.matmul(Q_heads, K_heads.transpose(-2, -1))    # shape: [batch_size, dec_len, enc_len]

        score = score / (self.d_head ** 0.5)  # shape: [batch_size, num_heads, dec_length, enc_length]

        if enc_padding_mask is not None:
            pad_mask = enc_padding_mask.unsqueeze(1).unsqueeze(2) # shape: [batch_size, 1, 1, enc_len]
            score = score.masked_fill(pad_mask, float('-inf')) # shape: [batch_size, num_heads, dec_len, enc_len]

        attention_weight = F.softmax(score, dim=-1)  # shape: [batch_size, num_heads, dec_len, enc_len]
        attention_weight = self.dropout_attn(attention_weight)

        attention = torch.matmul(attention_weight, V_heads) # shape: [batch_size, num_heads, dec_length, d_head]

        attention = attention.transpose(1,2).reshape(batch_size, dec_len, self.d_model) # shape: [batch_size, dec_length, d_model]

        output = self.O_W(attention)    # shape: [batch_size, dec_length, d_model]
        output = self.dropout_out(output)
        return output

