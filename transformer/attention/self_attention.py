import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, masked=False, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.masked = masked

        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.d_head = d_model // num_heads   # dimension of each head

        self.Q_W = nn.Linear(d_model, d_model, bias=False) # Query matrix
        self.K_W = nn.Linear(d_model, d_model, bias=False)  # Key matrix
        self.V_W = nn.Linear(d_model, d_model, bias=False)  # Value matrix

        # final Output layer
        self.O_W = nn.Linear(d_model, d_model, bias=False)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        batch_size, seq_length, _ = x.shape
        device = x.device

        Q = self.Q_W(x)  # shape: [batch_size, seq_length, d_model]
        K = self.K_W(x)  # shape: [batch_size, seq_length, d_model]
        V = self.V_W(x) # shape: [batch_size, seq_length, d_model]

        def split_head(x):
            # transpose: batch_size, seq_length, num_heads, d_head -> batch_size, num_heads, seq_length, d_head
            return x.reshape(batch_size, seq_length, self.num_heads, self.d_head).transpose(1,2)

        Q_heads = split_head(Q) # shape: [batch_size, num_heads, seq_length, d_head]
        K_heads = split_head(K) # shape: [batch_size, num_heads, seq_length, d_head]
        V_heads = split_head(V) # shape: [batch_size, num_heads, seq_length, d_head]

      # attention calculation
        score = torch.matmul(Q_heads, K_heads.transpose(-2, -1))    # shape: [batch_size, num_heads, seq_length(query), seq_len(key)]
        score = score / (self.d_head ** 0.5)  # shape: [batch_size, num_heads, seq_length, seq_len]
        if self.masked:
            masking = torch.tril(torch.ones(seq_length, seq_length)).to(device) # [seq_len, seq_len]
            masking = masking.unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, seq_len]
            score = score.masked_fill(masking==0, float('-inf')) # [batch_size, num_heads, seq_len, seq_len]

        if padding_mask is not None:
            pad_mask = padding_mask.unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, seq_len]
            score = score.masked_fill(pad_mask, float('-inf')) # [batch_size, num_heads, seq_len, seq_len]

        attention_weight = F.softmax(score, dim=-1) # [batch_size, num_heads, seq_len, seq_len]
        attention_weight = self.dropout_attn(attention_weight)

        attention = torch.matmul(attention_weight, V_heads) # shape: [batch_size, num_heads, seq_length, d_head]

        attention = attention.transpose(1,2).reshape(batch_size, seq_length, self.d_model) # shape: [batch_size, seq_length, d_model]

        output = self.O_W(attention)    # shape: [batch_size, seq_length, d_model]
        output = self.dropout_out(output)
        return output

