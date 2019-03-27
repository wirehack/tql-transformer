import sys
sys.path.append("/home/shuyanzh/workshop/tql-transformer/")
import time
import functools
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from src.utils.util_func import *
from copy import deepcopy
from src.data_loader import DataLoader, Batch
from src.config import argps
from src.train import make_model
import pickle
from collections import  defaultdict
import sentencepiece as spm

print = functools.partial(print, flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_data_iter(test_file, src_suffix, trg_suffix, w2i_src, w2i_trg):
    # read data
    src_file = open(test_file + "." + src_suffix, "r", encoding="utf-8")
    trg_file = open(test_file + "." + trg_suffix, "r", encoding="utf-8")
    line_tot = 0
    for src_line, trg_line in zip(src_file, trg_file):
        line_tot += 1
        src_tks = src_line.strip().split()
        trg_tks = trg_line.strip().split()
        src = [w2i_src[tk] for tk in src_tks + [w2i_src["</s>"]]]
        trg = [w2i_trg[tk] for tk in [w2i_trg["<s>"]] + trg_tks + [w2i_trg["</s>"]]]
        src = torch.LongTensor([src])
        trg = torch.LongTensor([trg])
        yield(Batch(src, trg))
    src_file.close()
    trg_file.close()
    print("[INFO] total test {:d}".format(line_tot))

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).long().to(device)
    for i in range(max_len-1):
        trg_mask = generate_subseq_mask(ys.size(1)).to(device)
        out = model.decode(memory, ys,
                           src_mask, trg_mask)
        prob = model.projector(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        next_word_ys = torch.ones(1, 1).fill_(next_word).long().to(device)
        ys = torch.cat([ys, next_word_ys], dim=1)
        if next_word == end_symbol:
            break
    return ys

def remove_unk_pairs(w2i_map:dict):
    unk_idx = w2i_map["<unk>"]
    new_map = {}
    for k, v in w2i_map.items():
        if v == unk_idx and k != "<unk>":
            continue
        else:
            new_map[k] = v
    if "<s>" in new_map:
        print("pad", new_map["<pad>"], "<s>", new_map[new_map["<s>"]], "</s>", new_map[new_map["</s>"]], "<unk>", new_map["<unk>"])
    else:
        print("pad", new_map["<pad>"], "</s>", new_map[new_map["</s>"]], "<unk>", new_map["<unk>"])
    return new_map

def load_sp_model(model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

def test(args):
    max_len = args.max_len
    w2i_src_file = args.w2i_map_file + "_src.pkl"
    w2i_trg_file = args.w2i_map_file + "_trg.pkl"
    with open(w2i_src_file, "rb") as f:
        w2i_src = pickle.load(f)
        w2i_src = remove_unk_pairs(w2i_src)
        src_vocab_size = len(w2i_src)
        print("[INFO] source vocab size: {:d}".format(src_vocab_size))
    with open(w2i_trg_file, "rb") as f:
        w2i_trg = pickle.load(f)
        w2i_trg = remove_unk_pairs(w2i_trg)
        trg_vocab_size = len(w2i_trg)
        i2w_trg = {v: str(k) for k, v in w2i_trg.items()}
        # FORCE TO FIX BUG
        i2w_trg[w2i_trg[w2i_trg["<s>"]]] = "<s>"
        i2w_trg[w2i_trg[w2i_trg["</s>"]]] = "</s>"
        print("[INFO] target vocab size: {:d}".format(trg_vocab_size))

    w2i_src = defaultdict(lambda: w2i_src["<unk>"], w2i_src)
    w2i_trg = defaultdict(lambda: w2i_trg["<unk>"], w2i_trg)

    # load model
    model_info = torch.load(args.model_path + "_" + str(args.model_ckpt) + ".tar")
    transformer = make_model(src_vocab_size, trg_vocab_size)
    transformer.load_state_dict(model_info["model_state_dict"])
    transformer.to(device)
    print("[INFO] reload model from {}".format(args.model_path + "_" + str(args.model_ckpt) + ".tar"))

    # load sentence piece model
    sp = load_sp_model(args.sp_model_path)

    # write result here
    with open(args.result_file, "w+", encoding="utf-8") as f:
        data_iter = test_data_iter(args.test_file, args.src_suffix, args.trg_suffix, w2i_src, w2i_trg)
        for idx, cur_sample in enumerate(data_iter):
            decoded = greedy_decode(transformer, cur_sample.src, cur_sample.src_mask,
                                    max_len, w2i_trg[w2i_trg["<s>"]], w2i_trg[w2i_trg["</s>"]])
            decoded_str = list(decoded.cpu().numpy()[0])
            decoded_str = [i2w_trg[x] for x in decoded_str]
            # remove <s>
            decoded_str = decoded_str[1:]
            # remove </s>
            if decoded_str[-1] == "</s>":
                decoded_str = decoded_str[:-1]
            decoded_str = sp.DecodePieces(decoded_str)
            f.write(decoded_str + "\n")
            if (idx + 1) % 1000 == 0:
                print(idx)




if __name__ == "__main__":
    args = argps()
    test(args)

