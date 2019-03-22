import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np

from src.sublayer import SublayerConnection, LayerNorm

# contains 3 sublayers: encoder_att, self_att, ff
class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, src_attn, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.src_attn = src_attn
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.Sequential(*[SublayerConnection(hidden_size, dropout) for i in range(3)])
        self.hidden_size = hidden_size
    def forward(self, input, memory, src_mask, trg_mask):
        self_attn_res = self.sublayer[0](input, self.self_attn(input, input, input, trg_mask))
        src_attn_res = self.sublayer[1](self_attn_res, self.src_attn(self_attn_res, memory, memory, src_mask))
        ff_res = self.sublayer[2](src_attn_res, self.feed_forward(src_attn_res))
        return ff_res

def generate_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape)).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Decoder(nn.Module):
    def __init__(self, N, decoder_layer: DecoderLayer):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(*[deepcopy(decoder_layer) for i in range(N)])
        self.layer_norm = LayerNorm(decoder_layer.hidden_size)

    def forward(self, input, memory, src_mask, trg_mask):
        for layer in self.layers:
            input = layer(input, memory, src_mask, trg_mask)
        layer_norm = self.layer_norm(input)
        return layer_norm
