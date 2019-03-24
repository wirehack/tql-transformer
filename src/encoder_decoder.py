'''
in this part, we implement encoder
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

class Projector(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Projector, self).__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input):
        '''
        :param input: [batch, len, hidden_size]
        :return: [batch, len, vocab_size]
        '''
        linear_res = self.linear(input)
        soft_max = F.softmax(linear_res, dim=-1)
        return soft_max

class LayerNorm(nn.Module):
    def __init__(self, hidden_size):
        super(LayerNorm, self).__init__()
        # self.a_2 = torch.ones(hidden_size, requires_grad=True)
        # self.b_2 = torch.zeros(hidden_size, requires_grad=True)
        self.a_2 = nn.Parameter(torch.ones(hidden_size))
        self.b_2 = nn.Parameter(torch.zeros(hidden_size))
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

    def forward(self, input: torch.Tensor, sublayer:nn.Module):
        # TODO why norm inside
        residual = input + self.dropout(sublayer(self.layer_norm(input)))
        return residual

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
        att_res = self.sublayer[0](input, lambda x: self.self_attn(x, x, x, mask))
        ff_res = self.sublayer[1](att_res, self.feed_forward)

        return ff_res

# a stack of N encoder layers
class Encoder(nn.Module):
    def __init__(self, encoder_layer:EncoderLayer, N):
        super(Encoder, self).__init__()
        self.layer_norm = LayerNorm(encoder_layer.hidden_size)
        # TODO maybe problematic
        self.layers = nn.Sequential(*[deepcopy(encoder_layer) for i in range(N)])

    def forward(self, input, mask):
        for layer in self.layers:
            input = layer(input, mask)
        norm_res = self.layer_norm(input)
        return norm_res


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
        self_attn_res = self.sublayer[0](input, lambda x: self.self_attn(x, x, x, trg_mask))
        src_attn_res = self.sublayer[1](self_attn_res, lambda x: self.src_attn(x, memory, memory, src_mask))
        ff_res = self.sublayer[2](src_attn_res, self.feed_forward)
        return ff_res

class Decoder(nn.Module):
    def __init__(self, decoder_layer: DecoderLayer, N):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(*[deepcopy(decoder_layer) for i in range(N)])
        self.layer_norm = LayerNorm(decoder_layer.hidden_size)

    def forward(self, input, memory, src_mask, trg_mask):
        for layer in self.layers:
            input = layer(input, memory, src_mask, trg_mask)
        layer_norm = self.layer_norm(input)
        return layer_norm

class EncoderDecoder(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed, trg_embed, projector):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.projector = projector

    def forward(self, src, trg, src_mask, trg_mask):
        encode_embed = self.src_embed(src)
        encode_res = self.encoder(encode_embed, src_mask)
        decode_embed = self.trg_embed(trg)
        decode_res = self.decoder(decode_embed, memory=encode_res, src_mask=src_mask, trg_mask=trg_mask)
        trg_prob = self.projector(decode_res)
        return trg_prob
