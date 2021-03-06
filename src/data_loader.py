import os
import sys
import random
from collections import defaultdict
import pickle
import torch
from src.utils.util_func import *

MAX_LEN = 70
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Batch:
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
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.token_num = torch.sum((self.trg_y != pad)).item()
        self.trg_y_mask = (self.trg_y != pad).float()
        self.src, self.src_mask = self.src.to(device), self.src_mask.to(device)
        self.trg, self.trg_mask = self.trg.to(device), self.trg_mask.to(device)
        self.trg_y = self.trg_y.to(device)  # self.trg_y_mask.to(device)

    @staticmethod
    def make_std_mask(trg, pad):
        # [batch_size, 1, len]
        trg_mask = (trg != pad).unsqueeze(-2)
        # [1, len, len]
        subseq_mask = subsequent_mask(trg.size(-1))
        # [batch_size, len, len]
        trg_mask = trg_mask & subseq_mask
        trg_mask = trg_mask.long()
        return trg_mask

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
        st = self.w2i_trg["<s>"]
        ed = self.w2i_trg["</s>"]
        unk = self.w2i_src["<unk>"]
        unk = self.w2i_trg["<unk>"]
        self.all_train = list(self.load_data(self.train_file, src_suffix, trg_suffix))
        self.w2i_src = defaultdict(lambda: self.w2i_src["<unk>"], self.w2i_src)
        self.w2i_trg = defaultdict(lambda: self.w2i_trg["<unk>"], self.w2i_trg)
        # sort training data by input length
        self.src_vocab_size = len(self.w2i_src)
        self.trg_vocab_size = len(self.w2i_trg)
        # save map
        with open(map_file + "_src.pkl", "wb") as f:
            pickle.dump(dict(self.w2i_src), f)
            print(
                "[INFO] save source char to idx map to :{}, len: {:d}".format(map_file + "_src.pkl", len(self.w2i_src)))
        with open(map_file + "_trg.pkl", "wb") as f:
            pickle.dump(dict(self.w2i_trg), f)
            print(
                "[INFO] save target char to idx map to :{}, len: {:d}".format(map_file + "_trg.pkl", len(self.w2i_trg)))

        if self.dev_file:
            self.all_dev = list(self.load_data(self.dev_file, src_suffix, trg_suffix))
        else:
            self.all_dev = None

    def load_data(self, file_name, src_suffix, trg_suffix):
        line_tot = 0
        src_file = open(file_name + "." + src_suffix, "r", encoding="utf-8")
        trg_file = open(file_name + "." + trg_suffix, "r", encoding="utf-8")
        for src_line, trg_line in zip(src_file, trg_file):
            line_tot += 1
            src_tks = src_line.strip().split()
            trg_tks = trg_line.strip().split()
            # filter sentence pair if ENG sentence is longer than 70 or FR > 80
            if len(trg_tks) >= MAX_LEN or len(src_tks) > MAX_LEN:
                continue
            src = [self.w2i_src[tk] for tk in src_tks]
            trg = [self.w2i_trg[tk] for tk in ["<s>"] + trg_tks + ["</s>"]]
            yield (src, trg)
        print("[INFO] number of lines in {}: {}".format(file_name, str(line_tot)))

        src_file.close()
        trg_file.close()

    # TODO change to token wise batch
    def create_batches(self, dataset: str):
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
            cur_pool = data[i:i + cur_size]
            cur_pool.sort(key=lambda x: (len(x[0]), len(x[1])))

            batches = []
            src_seq_len = [len(x[0]) for x in cur_pool]
            max_seq_len = src_seq_len[0]

            trg_seq_len = [len(x[1]) for x in cur_pool]
            max_trg_seq_len = trg_seq_len[0]

            prev_start = 0
            batch_size = 1

            for idx in range(1, len(src_seq_len)):
                cur_seq_len = src_seq_len[idx]
                max_seq_len = max(max_seq_len, cur_seq_len)
                new_tot_tokens = (batch_size + 1) * max_seq_len

                cur_trg_seq_len = trg_seq_len[idx]
                max_trg_seq_len = max(max_trg_seq_len, cur_trg_seq_len)
                new_trg_tot_tokens = (batch_size + 1) * max_trg_seq_len

                if new_tot_tokens > self.batch_size or new_trg_tot_tokens > self.batch_size \
                        or idx == (len(src_seq_len) - 1):
                    batches.append((prev_start, batch_size))
                    prev_start = idx
                    batch_size = 1
                else:
                    batch_size += 1
            batches.append((prev_start, batch_size))

            random.shuffle(batches)
            for st, batch_size in batches:
                cur_data = cur_pool[st:st + batch_size]
                src_sents = [torch.LongTensor(x[0]) for x in cur_data]
                trg_sents = [torch.LongTensor(x[1]) for x in cur_data]
                # pad
                src_sents = torch.nn.utils.rnn.pad_sequence(src_sents, batch_first=True, padding_value=self.pad)
                trg_sents = torch.nn.utils.rnn.pad_sequence(trg_sents, batch_first=True, padding_value=self.pad)

                yield (Batch(src_sents, trg_sents))

