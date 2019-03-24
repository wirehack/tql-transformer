import argparse
import os
import copy

def str2bool(s):
    if s == "0":
        return False
    else:
        return True

def argps():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="")
    parser.add_argument("--dev_file", default="")
    parser.add_argument("--src_suffix", default="fr")
    parser.add_argument("--trg_suffix", default="en")
    parser.add_argument("--w2i_map_file", default="")

    parser.add_argument("--model_path", default="")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pool", type=int, default=50000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--factor", type=int, default=2)

    args, _ = parser.parse_known_args()
    return args