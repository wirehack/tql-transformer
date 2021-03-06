import numpy as np
import torch
from torch import nn
import time
from src.utils.util_func import *
from src.modules import PositionwiseFeedForward, PositionalEncoding, Embeddings, LabelSmoothing, NoamOpt
from src.data_loader import DataLoader, Batch
from src.config import argps
from src.train import make_model, run_epoch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH_CHECK = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_gen(batch, nbatches):
    "Generate random data for a src-tgt copy task."
    V = 11
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = data
        tgt = data
        yield (Batch(src, tgt, 0))

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol)
    for i in range(max_len-1):
        ys = ys.long()
        out = model.decode(memory, ys,
                           src_mask, subsequent_mask(ys.size(1)))
        prob = model.projector(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def train_synthetic():
    V = 11
    model = make_model(V, V, N=2)
    model.to(device)
    criterion = LabelSmoothing(smoothing=0.0, vocab_size=V, pad_idx=0, device=device)
    optimizer = NoamOpt(model.parameters(), 512, 400, 1)
    for ep in range(1000):
        model.train()
        train_iter = data_gen(30, 20)
        run_epoch(ep, train_iter, model, criterion, optimizer)

        if (ep + 1) % EPOCH_CHECK == 0:
            print("===============eval===============")
            with torch.no_grad():
                model.eval()
                run_epoch(ep, data_gen(30, 20), model, criterion, optimizer=None)
                src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
                src_mask = torch.ones(1, 1, 10)
                print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
            print("===============eval===============")



if __name__ == "__main__":
    args = argps()
    train_synthetic()
