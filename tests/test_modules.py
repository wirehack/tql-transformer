import unittest
import torch.nn as nn
import torch.nn.functional as F
import torch
from src import modules
import math


class PositionwiseFeedForwardTrue(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForwardTrue, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EmbeddingsTrue(nn.Module):
    def __init__(self, d_model, vocab):
        super(EmbeddingsTrue, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncodingTrue(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncodingTrue, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class NoamOptTrue:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        # return rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothingTrue(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothingTrue, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        true_dist.require_grad = False
        return self.criterion(x, true_dist)


class TestModule(unittest.TestCase):
    def test_PFF(self):
        torch.manual_seed(0)
        myPFF = modules.PositionwiseFeedForward(512, 10, 0)
        torch.manual_seed(0)
        truePFF = PositionwiseFeedForwardTrue(512, 10, 0)
        rand_tensor = torch.rand(5, 512)
        my_output = myPFF(rand_tensor)
        true_output = truePFF(rand_tensor)
        self.assertTrue(torch.allclose(my_output, true_output))

    def test_embed(self):
        torch.manual_seed(0)
        my_emb = modules.Embeddings(512, 15)
        torch.manual_seed(0)
        true_emb = EmbeddingsTrue(512, 15)
        rand_tensor = torch.randint(0, 13, (5, 20))
        my_output = my_emb(rand_tensor)
        true_output = true_emb(rand_tensor)
        self.assertTrue(torch.allclose(my_output, true_output))

    def test_pe(self):
        torch.manual_seed(0)
        my_pe = modules.PositionalEncoding(512, 100, 0)
        torch.manual_seed(0)
        true_pe = PositionalEncodingTrue(512, 0, 100)
        rand_tensor = torch.rand(5, 100, 512)
        my_output = my_pe(rand_tensor)
        true_output = true_pe(rand_tensor)
        self.assertTrue(torch.allclose(my_output, true_output))

    def testOptim(self):
        rand_tensors = [torch.rand(5, 100, 512)]
        rand_tensors[0].require_grad = True
        torch.manual_seed(0)
        my_optim = modules.NoamOpt(rand_tensors, 512, 4000, 2)
        torch.manual_seed(0)
        true_optim = NoamOptTrue(512, 2, 4000,
                                 torch.optim.Adam(rand_tensors, lr=0, betas=(0.9, 0.98), eps=1e-9))
        for _ in range(10000):
            lr1 = my_optim.step()
            lr2 = true_optim.step()
            self.assertEqual(lr1, lr2)

    def test_lsm(self):
        predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                     [0, 0.2, 0.7, 0.1, 0],
                                     [0, 0.2, 0.7, 0.1, 0]])
        target = torch.LongTensor([2, 1, 0])
        torch.manual_seed(0)
        my_lsm = modules.LabelSmoothing(0.1, 5, 0)
        torch.manual_seed(0)
        true_lsm = LabelSmoothingTrue(5, 0, 0.1)
        my_output = my_lsm(predict, target)
        true_output = true_lsm(predict, target)
        self.assertTrue(torch.allclose(my_output, true_output))


if __name__ == '__main__':
    unittest.main()
