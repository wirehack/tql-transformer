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
    parser.add_argument("--train_file", default="/projects/tir2/users/shuyanzh/mtnt/data/clean.tok.bpe")
    parser.add_argument("--dev_file", default="/projects/tir2/users/shuyanzh/mtnt/data/dev.tok.bpe")
    parser.add_argument("--src_suffix", default="fr")
    parser.add_argument("--trg_suffix", default="en")
    parser.add_argument("--w2i_map_file", default="/projects/tir2/users/shuyanzh/mtnt/model/fr_en_clean_map")

    parser.add_argument("--model_path", default="/projects/tir2/users/shuyanzh/mtnt/model/fr_en_clean_model")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pool_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)


    args, _ = parser.parse_known_args()
    return args