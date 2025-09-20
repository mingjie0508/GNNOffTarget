#!/usr/bin/env python
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.cfg import RunConfig
from src.builders import create_model
from src.trainer import Trainer
from src.utils.misc_utils import seed_everything


def main():
    # Configs
    # TODO, add optional argparse overwrite
    cfg = RunConfig()
    cfg.paths.out_dir = Path("outputs/base")  # you can override manually
    cfg.train.epochs = 5                      # quick hackathon run
    cfg.model.name = "gcn"                    # or "gat"

    seed_everything(cfg.seed)

    # TODO
    # Datasets and Dataloaders
    

    # Model + Loss
    model = create_model(cfg)
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    trainer = Trainer(cfg, model, train_loader, val_loader, criterion=criterion)
    trainer.train()
    

if __name__ == "__main__":
    main()