#!/usr/bin/env python
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.cfg import RunConfig
from src.builders import create_model
from src.trainer import Trainer
from src.utils.misc_utils import seed_everything
from torch_geometric.data import Data

import argparse
import numpy as np
import pandas as pd
import json

from src.dataset import CRISPRoffTDataset
from src.utils.data_utils import get_splits


def main(args):

    # Configs
    cfg = RunConfig()
    cfg.data.guide_seq_path = Path(args.guide_seq_path)
    cfg.data.target_seq_path = Path(args.target_seg_path)
    cfg.data.metadata_path = Path(args.metadata_path)
    cfg.data.vocab_path = Path(args.vocab_path) if args.vocab_path else None

    cfg.train.device = args.device
    cfg.paths.output_dir = Path(args.output)

    seed_everything(cfg.seed)

    # Data
    full = CRISPRoffTDataset(cfg, idxs=None)
    train_idx, val_idx, test_idx = get_splits(len(full.groups), full.groups, seed=cfg.seed)

    vocabs = {}
    if cfg.data.vocab_path is not None:
        with open(cfg.data.vocab_path, "r") as f:
            vocabs = json.load(f)

    train_ds = CRISPRoffTDataset(cfg, train_idx, vocabs=vocabs)
    val_ds = CRISPRoffTDataset(cfg, val_idx, vocabs=train_ds.vocabs)
    test_ds = CRISPRoffTDataset(cfg, test_idx, vocabs=train_ds.vocabs)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=True)

    # Model + Loss
    model = create_model(cfg)
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    trainer = Trainer(cfg, model, train_loader, val_loader, criterion=criterion)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--guide_seq_path", type=str)
    parser.add_argument("--target_seq_path", type=str)
    parser.add_argument("--metadata_path", type=str)
    parser.add_argument("--vocab_path", type=str, default=None)

    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="output")

    args = parser.parse_args()

    main(args)
