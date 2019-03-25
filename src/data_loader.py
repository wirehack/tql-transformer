import os
import sys
import random
from collections import defaultdict
import pickle
import torch
from src.utils.util_func import *

class Batch():
    def __init__(self, src:torch.Tensor, trg:torch.Tensor=None, pad=0):
        '''
        :param src: [batch_size, len]
        :param trg: [batch_size, len]
        :param pad: int
        '''
        self.src = src
        # [batch_size, 1, len]
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            # [batch_size, len - 1, len - 1]
            self.trg_mask = self.create_trg_mask(self.trg, pad)
        self.token_num = torch.sum(self.trg_y != pad)

    def create_trg_mask(self, input:torch.Tensor, pad):
        # [batch_size, 1, len]
        mask = (input != pad).unsqueeze(-2)
        # [1, len, len]
        subseq_mask = generate_subseq_mask(input.size(1))
        # [batch_size, len, len]
        mask =  mask.long() & subseq_mask.long()

        return mask

class DataLoader:
    def __init__(self, train_file, dev_file, src_suffix, trg_suffix, map_file, batch_size, pool_size, pad=0):
        self.batch_size = batch_size
        self.pad = pad
        self.pool_size = pool_size
        self.train_file = train_file
        self.dev_file = dev_file
        self.w2i_src = defaultdict(lambda: len(self.w2i_src))
        self.w2i_trg = defaultdict(lambda: len(self.w2i_trg))
        pad = self.w2i_src["<pad>"]
        pad = self.w2i_trg["<pad>"]
        ed = self.w2i_src["</s>"]
        st = self.w2i_trg["<s>"]
        ed = self.w2i_trg["</s>"]
        unk = self.w2i_src["<unk>"]
        unk = self.w2i_trg["<unk>"]
        self.all_train = list(self.load_data(self.train_file, src_suffix, trg_suffix))
        self.w2i_src = defaultdict(lambda: self.w2i_src["<unk>"], self.w2i_src)
        self.w2i_trg = defaultdict(lambda: self.w2i_trg["<unk>"], self.w2i_trg)
        self.i2w_src = {v: k for k, v in self.w2i_src.items()}
        self.i2w_trg = {v: k for k, v in self.w2i_trg.items()}
        # sort training data by input length
        self.src_vocab_size = len(self.w2i_src)
        self.trg_vocab_size = len(self.w2i_trg)

        if self.dev_file:
            self.all_dev = list(self.load_data(self.dev_file, src_suffix, trg_suffix))
        else:
            self.all_dev = None

        # save map
        with open(map_file + "_src.pkl", "wb") as f:
            pickle.dump(dict(self.w2i_src), f)
            print("[INFO] save source char to idx map to :{}, len: {:d}".format(map_file + "_src.pkl", len(self.w2i_src)))
        with open(map_file + "_trg.pkl", "wb") as f:
            pickle.dump(dict(self.w2i_trg), f)
            print("[INFO] save target char to idx map to :{}, len: {:d}".format(map_file + "_trg.pkl", len(self.w2i_trg)))


    def load_data(self, file_name, src_suffix, trg_suffix):
        line_tot = 0
        src_file = open(file_name + "." + src_suffix, "r", encoding="utf-8")
        trg_file = open(file_name + "." + trg_suffix, "r", encoding="utf-8")
        for src_line, trg_line in zip(src_file, trg_file):
            line_tot += 1
            src_tks = src_line.strip().split()
            trg_tks = trg_line.strip().split()
            src = [self.w2i_src[tk] for tk in src_tks +  [self.w2i_src["</s>"]]]
            trg = [self.w2i_trg[tk] for tk in [self.w2i_trg["<s>"]] + trg_tks + [self.w2i_trg["</s>"]]]
            yield(src, trg)
        print("[INFO] number of lines in {}: {}".format(file_name, str(line_tot)))

        src_file.close()
        trg_file.close()


    def create_batches(self, dataset:str):
        if dataset == "train":
            data = self.all_train
            random.shuffle(data)
        elif dataset == "dev":
            data = self.all_dev
        else:
            raise NotImplementedError
        # make pools
        for i in range(0, len(data), self.pool_size):
            cur_size = min(self.pool_size, len(data) - i)
            cur_pool = data[i:i+cur_size]
            cur_pool.sort(key=lambda x: len(x[0]))
            for j in range(0, len(cur_pool), self.batch_size):
                cur_size = min(self.batch_size, len(cur_pool) - j)
                cur_data = cur_pool[j:j+cur_size]
                src_sents = [x[0] for x in cur_data]
                src_seq_len = [len(x) for x in src_sents]
                src_max_len = max(src_seq_len)

                trg_sents = [x[1] for x in cur_data]
                trg_seq_len = [len(x) for x in trg_sents]
                trg_max_len = max(trg_seq_len)
                # pad
                src_sents = torch.LongTensor([x + [self.pad for i in range(src_max_len-x_len)] for x, x_len in zip(src_sents, src_seq_len)])
                trg_sents = torch.LongTensor([x + [self.pad for i in range(trg_max_len-x_len)] for x, x_len in zip(trg_sents, trg_seq_len)])
                yield(Batch(src_sents, trg_sents))
