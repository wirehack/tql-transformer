import torch.nn as nn
import torch
import math


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_probs):
        super(PositionwiseFeedForward, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_probs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.mul = math.sqrt(d_model)
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * self.mul


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout_probs, device):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32)
        pos = pos.unsqueeze(1)
        exponent = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(pos * exponent)
        self.pe[:, 1::2] = torch.cos(pos * exponent)
        self.pe = self.pe.unsqueeze(0)
        self.pe.requires_grad = False
        self.pe = self.pe.to(device)
        self.dropout = nn.Dropout(dropout_probs)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class NoamOpt:
    def __init__(self, params, d_model, warmup, factor):
        self.optimizer = torch.optim.Adam(params, lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.warmup_mul = math.pow(warmup, -1.5)
        self.lr = 0
        self.num_steps = 0
        self.d_model = d_model
        self.factor = factor  # TODO: check this

    def step(self):
        self.num_steps += 1
        lr = math.pow(self.d_model, -0.5) * min(math.pow(self.num_steps, -0.5), self.num_steps * self.warmup_mul)
        lr *= self.factor
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()
        # return lr

    def zero_grad(self):
        self.optimizer.zero_grad()


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing, vocab_size, pad_idx, device):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.device = device

    def forward(self, x, target):
        '''
        :param x: [batch, len, vocab_size]
        :param target: [batch, len]
        :return:
        '''
        x = x.contiguous().view(-1, x.size(-1))
        target = target.contiguous().view(-1)
        smooth_target = torch.zeros(x.size()).to(self.device)
        smooth_target.fill_(self.smoothing / (x.size(1) - 2))  # ignore s and eos
        smooth_target.scatter_(-1, target.unsqueeze(1), 1 - self.smoothing)
        smooth_target[:, self.pad_idx] = 0
        mask = torch.nonzero(target == self.pad_idx)
        if mask.dim() > 0:
            smooth_target.index_fill_(0, mask.squeeze(), 0)
        smooth_target.requires_grad = False
        return self.criterion(x, smooth_target)
