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

    tokenizer = AutoTokenizer.from_pretrained(
        EMBED_MODEL_NAME, local_files_only=False, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        EMBED_MODEL_NAME, local_files_only=False, trust_remote_code=True
    )

    model.to(args.device)

    df = pd.read_csv(args.dataset_path)
    guide_seg = list(df["guide_sequence"])
    target_seq = list(df["target_sequence"])

    guide_emb = []
    target_emb = []

    for seq in tqdm(guide_seg, desc="Embedding Guide Sequence"):
        inputs = tokenizer(seq, return_tensors="pt")["input_ids"]
        inputs = inputs.to(args.device)
        hidden_states = model(inputs)[0]

        embedding_mean = torch.mean(hidden_states[0], dim=0)
        guide_emb.append(embedding_mean.detach().cpu().numpy())

    for seq in tqdm(target_seq, desc="Embedding Target Sequence"):
        inputs = tokenizer(seq, return_tensors="pt")["input_ids"]
        inputs = inputs.to(args.device)
        hidden_states = model(inputs)[0]

        embedding_mean = torch.mean(hidden_states[0], dim=0)
        target_emb.append(embedding_mean.detach().cpu().numpy())

    os.makedirs(args.output, exist_ok=True)
    np.save(Path(args.output) / "guide_emb.npy", np.array(guide_emb).astype("float32"))
    np.save(Path(args.output) / "target_emb.npy", np.array(target_emb).astype("float32"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="seq_embeddings")

    args = parser.parse_args()

    main(args)
