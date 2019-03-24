import unittest
import torch.nn as nn
import torch.nn.functional as F
import torch
from src import encoder_decoder
from src import mul_attention
from src import modules
import math
import copy


class PositionwiseFeedForwardTrue(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForwardTrue, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttentionTrue(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionTrue, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayerTrue(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayerTrue, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def init_weight(model):
    for p in model.parameters():
        torch.manual_seed(0)
        nn.init.normal_(p, 10, 100)


class DecoderLayerTrue(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayerTrue, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class TestModule(unittest.TestCase):
    def test_encoder_layer(self):
        torch.manual_seed(0)
        myNorm = encoder_decoder.LayerNorm(8)
        myAttention = mul_attention.MultiHeadAttention(head_num=2, hidden_size=8, dropout=0)
        myFF = modules.PositionwiseFeedForward(d_model=8, d_ff=16, dropout_probs=0)
        myEncoderLayer = encoder_decoder.EncoderLayer(hidden_size=8, self_attn=myAttention, feed_forward=myFF,
                                                      dropout=0)
        init_weight(myEncoderLayer)

        torch.manual_seed(0)
        trueNorm = LayerNorm(8)
        trueAttention = MultiHeadedAttentionTrue(2, 8, 0)
        trueFF = PositionwiseFeedForwardTrue(8, 16, 0)
        trueEncoderLayer = EncoderLayerTrue(8, trueAttention, trueFF, 0)
        init_weight(trueEncoderLayer)

        rand_tensor = torch.randn(3, 10, 8)
        mask = torch.zeros(3, 10).long()
        mask[0, 0] = 1
        mask[1, 0] = 1
        mask[1, 1] = 1
        mask = mask.unsqueeze(1)
        my_norm, true_norm = myNorm(rand_tensor), trueNorm(rand_tensor)
        my_output = myEncoderLayer(rand_tensor, mask)
        true_output = trueEncoderLayer(rand_tensor, mask)

        self.assertTrue(torch.allclose(myAttention(rand_tensor, rand_tensor, rand_tensor, mask),
                                       trueAttention(rand_tensor, rand_tensor, rand_tensor, mask)))
        self.assertTrue(torch.allclose(my_norm, true_norm))
        self.assertTrue(torch.allclose(myFF(rand_tensor), trueFF(rand_tensor)))
        self.assertTrue(torch.allclose(my_output, true_output))

    def test_decoder_layer(self):
        torch.manual_seed(0)
        myAttention1 = mul_attention.MultiHeadAttention(head_num=2, hidden_size=8, dropout=0)
        myAttention2 = mul_attention.MultiHeadAttention(head_num=2, hidden_size=8, dropout=0)
        myFF = modules.PositionwiseFeedForward(d_model=8, d_ff=16, dropout_probs=0)
        myDecoderLayer = encoder_decoder.DecoderLayer(hidden_size=8, src_attn=myAttention1, self_attn=myAttention2,
                                                      feed_forward=myFF,
                                                      dropout=0)
        init_weight(myDecoderLayer)

        torch.manual_seed(0)
        trueAttention1 = MultiHeadedAttentionTrue(2, 8, 0)
        trueAttention2 = MultiHeadedAttentionTrue(2, 8, 0)
        trueFF = PositionwiseFeedForwardTrue(8, 16, 0)
        trueDecoderLayer = DecoderLayerTrue(8, trueAttention1, trueAttention2, trueFF, 0)
        init_weight(trueDecoderLayer)

        rand_tensor1 = torch.randn(3, 10, 8)
        rand_tensor2 = torch.rand(3, 10, 8)
        mask1 = torch.zeros(3, 10).long()
        mask1[0, 0] = 1
        mask1[1, 0] = 1
        mask1[1, 1] = 1
        mask1 = mask1.unsqueeze(1)
        my_output = myDecoderLayer(rand_tensor1, rand_tensor2, mask1, mask1)
        true_output = trueDecoderLayer(rand_tensor1, rand_tensor2, mask1, mask1)
        self.assertTrue(torch.allclose(my_output, true_output))


if __name__ == '__main__':
    unittest.main()
