import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d_model):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, 4*d_model)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(4*d_model, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x