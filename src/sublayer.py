import numpy as np
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_size):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.ones(hidden_size, requires_grad=True)
        self.b_2 = torch.ones(hidden_size, requires_grad=True)
    def forward(self, input:torch.Tensor):
        mean = torch.mean(input, dim=-1, keepdim=True)
        std = torch.std(input, dim=-1, keepdim=True)
        norm_res = self.a_2 * (input - mean) / (std + 1e-6) + self.b_2

        return norm_res

class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, sublayer_input):
        # residual connection
        # TODO different from original
        residual = input + sublayer_input
        norm_res = self.dropout(self.layer_norm(residual))
        return norm_res