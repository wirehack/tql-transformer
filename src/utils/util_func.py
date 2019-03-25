import numpy as np
import torch
def generate_subseq_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape)).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0