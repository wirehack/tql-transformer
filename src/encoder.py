'''
in this part, we implement encoder
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

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

    def forward(self, input: torch.Tensor, sublayer:nn.Module):
        # residual connection
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
    def __init__(self, N, decoder_layer: DecoderLayer):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(*[deepcopy(decoder_layer) for i in range(N)])
        self.layer_norm = LayerNorm(decoder_layer.hidden_size)

    def forward(self, input, memory, src_mask, trg_mask):
        for layer in self.layers:
            input = layer(input, memory, src_mask, trg_mask)
        layer_norm = self.layer_norm(input)
        return layer_norm

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, hidden_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % head_num == 0
        # number of hidden size after linear transform of EACH head
        self.d_k = int(hidden_size / head_num)
        self.head_num = head_num
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(3)])
        self.out_linear = nn.Linear(hidden_size, hidden_size)

    # calculate how likely the query and key, get w_i for each key_i. The result is sum_{w_i value_i}
    # the attention is calculated by DOT product
    def attention(self, query, key, value, mask: torch.Tensor = None, dropout: nn.Dropout = None):
        '''
        :param query: [batch_size, head_num, qlen, d_k]
        :param key: [batch_size, head_num, klen, d_k]
        :param value: [batch_size, head_num, klen, d_k]
        :param mask: [batch_size, 1, qlen, klen] or [batch_size, 1, 1, klen]
        :return: attention score: [batch_size, head_num, qlen, d_k]
        '''
        d_k = query.size(-1)
        # [batch_size, head_num, qlen, klen]
        matmul_res = torch.matmul(query, torch.transpose(key, -1, -2))
        norm_matmul = matmul_res / math.sqrt(d_k)
        if mask != None:
            norm_matmul = norm_matmul.masked_fill(mask == 0, -1e9)
        # softmax over key
        attn_scores = F.softmax(norm_matmul, dim=-1)

        if dropout != None:
            attn_scores = dropout(attn_scores)
        # [batch_size, head_num, qlen, d_k]
        attn_values = torch.matmul(attn_scores, value)

        return attn_scores, attn_values

    def forward(self, query, key, value, mask=None):
        '''
        :param query:[batch_size, len, hidden_size]
        :param key: [batch_size, len, hidden_size]
        :param value: [batch_size, len, hidden_size]
        :param mask: [batch_size, len, len] or [batch_size, 1, len]
        :return:
        '''
        batch_size = query.size(0)
        if mask != None:
            # the same for all h heads
            # [bz, h, len, len]
            mask = mask.unsqueeze(1)
        # do linear projections
        projects = []
        for input, linear in zip([query, key, value], self.linears):
            project = linear(input)
            project = project.view(batch_size, -1, self.head_num, self.d_k)
            project = torch.transpose(project, 1, 2)
            projects.append(project)
        pquery, pkey, pvalue = projects

        attn_values, self.attn_scores = self.attention(pquery, pkey, pvalue, mask=mask, dropout=self.dropout)
        # concatenate heads
        # [batch_size, qlen, head_num, d_k]
        attn_values = torch.transpose(attn_values, 1, 2).continuous()
        attn_values = attn_values.view(batch_size, -1, self.head_num * self.d_k)
        attn_values = self.out_linear(attn_values)

        return attn_values


class EncoderDecoder(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, src, trg, src_mask, trg_mask):
        encode_embed = self.src_embed(src)
        encode_res = self.encoder(encode_embed, src_mask)
        decode_embed = self.trg_embed(trg)
        decode_res = self.decoder(decode_embed, memory=encode_res, src_mask=src_mask, trg_mask=trg_mask)
        return decode_res
