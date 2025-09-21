import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path


def main(args):
    EMBED_MODEL_NAME = "zhihan1996/DNABERT-S"

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, local_files_only=False, trust_remote_code=True)
    model = AutoModel.from_pretrained(EMBED_MODEL_NAME, local_files_only=False, trust_remote_code=True)

    model.to(args.device)

    df = pd.read_csv(args.dataset_path)
    guide_seg = list(df["guide_sequence"])
    target_seq = list(df["target_sequence"])

    guide_emb = []
    target_emb = []

    for i in tqdm(range(0, len(guide_seg), args.batch_size), desc="Embedding Guide Sequence"):
        batch_seqs = guide_seg[i : i + args.batch_size]
        x = tokenizer(batch_seqs, return_tensors="pt", padding=True)["input_ids"].to(args.device)
        x = model(x)[0]
        embs = x.mean(dim=1)

        guide_emb.extend(embs.detach().cpu().numpy())

    for i in tqdm(range(0, len(target_seq), args.batch_size), desc="Embedding Target Sequence"):
        batch_seqs = target_seq[i : i + args.batch_size]
        x = tokenizer(batch_seqs, return_tensors="pt", padding=True)["input_ids"].to(args.device)
        x = model(x)[0]
        embs = x.mean(dim=1)

        target_emb.extend(embs.detach().cpu().numpy())

    os.makedirs(args.output, exist_ok=True)
    np.save(Path(args.output) / "guide_emb.npy", np.array(guide_emb).astype("float32"))
    np.save(Path(args.output) / "target_emb.npy", np.array(target_emb).astype("float32"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="seq_embeddings")

    args = parser.parse_args()

    main(args)
