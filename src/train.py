import numpy as np
import torch
from torch import nn
from torchtext import data, datasets
import time
import torch.nn.functional as F
from src.utils.util_func import *
from src.encoder_decoder import EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer, Projector
from src import PositionwiseFeedForward, PositionalEncoding, Embedding, LableSmoothing, NoamOpt
from src.attention import MultiHeadAttention
from copy import deepcopy
from src.data_loader import DataLoader

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH_CHECK=1
# make the model
def make_model(src_vocab_size, trg_vocab_size, N=6, d_model=512, d_ff=2048, head_num=8, dropout=0.1):
    attn = MultiHeadAttention(head_num, d_model, dropout)
    feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(feed_forward), dropout), N),
        Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(feed_forward), dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab_size), deepcopy(position)),
        nn.Sequential(Embedding(d_model, trg_vocab_size), deepcopy(position)),
        Projector(d_model, trg_vocab_size))
    # init with Xavier
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def calc_loss(criterion, optimizer, predict, gold, token_num):
    '''
    :param predict: [batch_size, len, trg_vocab_size]
    :param gold: [batch_size, len]
    :param token_num: int
    :return: loss of the batch, also update parameters
    '''
    # project to vocab size
    loss = self.criterion(predict, gold) / token_num
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    return loss * token_num


def train_epoch(data_iter, model:EncoderDecoder, loss_compute):
    start_time = time.time()
    tot_tokens = 0
    train_loss = 0
    tokens = 0
    for i, cur_batch in enumerate(data_iter):
        output = model(cur_batch.src, cur_batch.trg,
                       cur_batch.src_mask, cur_batch.trg_mask)
        loss = loss_compute(output, cur_batch.trg_y, cur_batch.token_num)
        train_loss += loss
        tot_tokens += cur_batch.token_num
        tokens += cur_batch.token_num

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print("[INFO] in epoch {:d}, loss: {:.4f}, tokens per sec: {:.4f}".format(i, loss / cur_batch.ntokens, tokens / elapsed))
            start_time = time.time()
            tokens = 0

    return train_loss / tot_tokens

def run_epoch(epoch, data_iter, model, criterion, optimizer=None):
    start_time = time.time()
    tot_tokens = 0
    train_loss = 0
    tokens = 0
    for i, cur_batch in enumerate(data_iter):
        output = model(cur_batch.src, cur_batch.trg,
                       cur_batch.src_mask, cur_batch.trg_mask)
        loss = criterion(output, cur_batch.trg_y) / cur_batch.token_num
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss += loss
        tot_tokens += cur_batch.token_num
        tokens += cur_batch.token_num
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print("[INFO] in epoch {:d}, loss: {:.4f}, tokens per sec: {:.4f}".format(i, loss / cur_batch.ntokens,
                                                                                      tokens / elapsed))
            start_time = time.time()
            tokens = 0
    print("[INFO] epoch {:d}: loss={:.1f}/{:.1f}={:.4f}, time={:.2f}".format(epoch,
                                                                    train_loss, tot_tokens, train_loss / tot_tokens,
                                                                    time.time() - start_time))
    return train_loss / tot_tokens

def save_model(model:nn.Module, path):
    torch.save({"model_state_dict":model.state_dict()}, path)
    print("[SAVE] save model!")

def train():
    data_loader = DataLoader()
    src_vocab_size, trg_vocab_size = data_loader.src_vocab_size, data_loader.trg_vocab_size
    model = make_model(src_vocab_size, trg_vocab_size)
    model.to(device)
    criterion = LableSmoothing()
    optimizer = NoamOpt()
    best_loss = float('inf')
    for ep in range(1000):
        model.train()
        train_iter = data_loader.create_batches("train")
        run_epoch(ep, train_iter, model, criterion, optimizer)

        if (ep + 1) % EPOCH_CHECK == 0:
            print("[EVALUATING]")
            with torch.no_grad():
                model.eval()
                dev_iter = data_loader.create_batches("dev")
                dev_loss = run_epoch(ep, dev_iter, model, criterion, optimizer=None)
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    save_model(model, model_path + "_" + str(ep) + ".tar")


