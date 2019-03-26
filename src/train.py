import sys

sys.path.append("/home/ubuntu/tql-transformer/")
import time
import functools
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from src.utils.util_func import *
from src.encoder_decoder import EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer, Projector
from src.modules import PositionwiseFeedForward, PositionalEncoding, Embeddings, LabelSmoothing, NoamOpt
from src.mul_attention import MultiHeadAttention
from copy import deepcopy
from src.data_loader import DataLoader
from src.config import argps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH_CHECK = 1
STEP_CHECK = 1000
print = functools.partial(print, flush=True)


# make the model
def make_model(src_vocab_size, trg_vocab_size, N=6, d_model=512, d_ff=2048, head_num=8, dropout=0.1):
    attn = MultiHeadAttention(head_num=head_num, hidden_size=d_model, dropout=dropout)
    feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_probs=dropout)
    position = PositionalEncoding(d_model, max_len=5000, dropout_probs=dropout, device=device)
    model = EncoderDecoder(
        Encoder(EncoderLayer(hidden_size=d_model, self_attn=deepcopy(attn), feed_forward=deepcopy(feed_forward),
                             dropout=dropout), N=N),
        Decoder(DecoderLayer(hidden_size=d_model, self_attn=deepcopy(attn), src_attn=deepcopy(attn),
                             feed_forward=deepcopy(feed_forward), dropout=dropout), N=N),
        nn.Sequential(Embeddings(d_model, src_vocab_size), deepcopy(position)),
        nn.Sequential(Embeddings(d_model, trg_vocab_size), deepcopy(position)),
        Projector(d_model, trg_vocab_size))
    # init with Xavier
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def run_epoch(epoch, data_iter, model, criterion, optimizer=None):
    epoch_start = time.time()
    start_time = time.time()

    tot_tokens = 0
    tot_loss = 0
    tot_ll_sum = 0

    record_tokens = 0
    record_loss = 0
    record_ll_sum = 0

    for i, cur_batch in enumerate(data_iter):
        # [batch_size, len, vocab_size]
        # trg_y [batch_size, len]
        output = model(cur_batch.src, cur_batch.trg,
                       cur_batch.src_mask, cur_batch.trg_mask)
        loss = criterion(output, cur_batch.trg_y) / cur_batch.token_num
        # output_pll = output * cur_batch.trg_y_mask.unsqueeze(-1)
        ll_sum = torch.IntTensor([0])
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss = loss * cur_batch.token_num
        tot_loss += loss.item()
        tot_ll_sum += ll_sum.item()
        tot_tokens += cur_batch.token_num

        record_loss += loss.item()
        record_ll_sum += ll_sum.item()
        record_tokens += cur_batch.token_num

        if i % STEP_CHECK == 1:
            elapsed = time.time() - start_time
            record_ppl = math.exp(record_ll_sum / record_tokens)
            # TODO why not / elapse work
            print(
                "[INFO] epoch step {:d}, loss: {:.6f}, perplexity: {:.4f}, time: {:.2f}s, tokens per sec: {:.2f}".format(
                    i, record_loss / record_tokens,
                    record_ppl, elapsed, record_tokens / (elapsed + 1)))
            start_time = time.time()
            record_loss = 0
            record_ll_sum = 0
            record_tokens = 0

    tot_ppl = math.exp(tot_ll_sum / tot_tokens)
    print("[INFO] epoch {:d}: loss={:.6f}, perplexity: {:.4f}, time={:.2f}".format(epoch,
                                                                                   tot_loss / tot_tokens,
                                                                                   tot_ppl,
                                                                                   time.time() - epoch_start))
    return tot_loss / tot_tokens


def save_model(model: nn.Module, path):
    torch.save({"model_state_dict": model.state_dict()}, path)
    print("[SAVE] save model!")


def train(args):
    data_loader = DataLoader(args.train_file, args.dev_file, args.src_suffix, args.trg_suffix,
                             args.w2i_map_file, args.batch_size, args.pool_size, pad=0)
    src_vocab_size, trg_vocab_size = data_loader.src_vocab_size, data_loader.trg_vocab_size
    model = make_model(src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size)
    model.to(device)
    criterion = LabelSmoothing(smoothing=0.1, vocab_size=trg_vocab_size, pad_idx=0, device=device)
    optimizer = NoamOpt(model.parameters(), d_model=args.d_model, warmup=args.warmup, factor=args.factor)
    # best_loss = float('inf')
    for ep in range(1000):
        model.train()
        train_iter = data_loader.create_batches("train")
        run_epoch(ep, train_iter, model, criterion, optimizer)

        if (ep + 1) % EPOCH_CHECK == 0:
            with torch.no_grad():
                model.eval()
                dev_iter = data_loader.create_batches("dev")
                print("===============eval===============")
                run_epoch(ep, dev_iter, model, criterion, optimizer=None)
                # if dev_loss < best_loss:
                # best_loss = dev_loss
                save_model(model, args.model_path + "_" + str(ep) + ".tar")
                print("===============eval===============")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = argps()
    train(args)
