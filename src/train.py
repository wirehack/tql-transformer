import numpy as np
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
        if trg:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            # [batch_size, len, len]
            self.trg_mask = self.create_trg_mask(trg, pad)
        self.token_num = torch.sum(self.trg_y != pad)

    def create_trg_mask(self, input:torch.Tensor, pad):
        # [batch_size, 1, len]
        mask = (input != pad).unsqueeze(-2)
        # [1, len, len]
        subseq_mask = generate_subseq_mask(input.size(1))
        # [batch_size, len, len]
        mask =  mask.long() & subseq_mask.long()

        return mask

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    # credit: http://nlp.seas.harvard.edu/2018/04/03/attention.html
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)