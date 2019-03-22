'''
in this part, we implement encoder
'''
import torch
import torch.nn as nn
from copy import deepcopy

from src.sublayer import SublayerConnection, LayerNorm


# two sublayers: multi-head attention and ff
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.Sequential(*[SublayerConnection(hidden_size, dropout) for i in range(2)])
        self.hidden_size = hidden_size

    def forward(self, input, mask):
        # multi head attention
        att_res = self.sublayer[0](input, self.self_attn(input, input, input, mask))
        ff_res = self.sublayer[1](att_res, self.feed_forward)

        return ff_res

# a stack of N encoder layers
class Encoder(nn.Module):
    def __init__(self, N, encoder_layer:EncoderLayer):
        super(Encoder, self).__init__()
        self.layer_norm = LayerNorm(encoder_layer.hidden_size)
        # TODO maybe problematic
        self.layers = nn.Sequential(*[deepcopy(encoder_layer) for i in range(N)])

    def forward(self, input, mask):
        for layer in self.layers:
            input = layer(input, mask)
        norm_res = self.layer_norm(input)
        return norm_res



