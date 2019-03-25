import numpy as np
import torch
from torch import nn
import time
from src.utils.util_func import *
from src.encoder_decoder import EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer, Projector
from src.modules import PositionwiseFeedForward, PositionalEncoding, Embeddings, LabelSmoothing, NoamOpt
from src.mul_attention import MultiHeadAttention
from copy import deepcopy
from src.data_loader import DataLoader, Batch
from src.config import argps
from src.train import make_model, run_epoch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH_CHECK = 1

def data_gen(batch, nbatches):
    "Generate random data for a src-tgt copy task."
    V = 11
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = data
        tgt = data
        yield (Batch(src, tgt, 0))


def train_synthetic():
    V = 11
    model = make_model(V, V)
    model.to(device)
    criterion = LabelSmoothing(smoothing=0.0, vocab_size=V, pad_idx=0)
    optimizer = NoamOpt(model.parameters(), args.d_model, args.warmup, args.factor)
    best_loss = float('inf')
    for ep in range(1000):
        model.train()
        train_iter = data_gen(30, 20)
        run_epoch(ep, train_iter, model, criterion, optimizer)

        if (ep + 1) % EPOCH_CHECK == 0:
            print("===============eval===============")
            with torch.no_grad():
                model.eval()
                dev_loss = run_epoch(ep, data_gen(30, 20), model, criterion, optimizer=None)
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    # save_model(model, args.model_path + "_" + str(ep) + ".tar")
            print("===============eval===============")



if __name__ == "__main__":
    args = argps()
    train_synthetic()
