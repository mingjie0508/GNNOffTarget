import argparse
import json
import os
from pathlib import Path

from torch.utils.data import DataLoader

from src.cfg import RunConfig
from src.builders import create_model
from src.training.trainer import Trainer
from src.utils.seed import seed_everything
from src.data.dataset import CRISPRoffTDataset
from src.data.split import get_splits
from src.data.transform import get_collate_fn, fit_num_scaler


def main(args):

    # Configs
    cfg = RunConfig()
    cfg.data.guide_seq_path = Path(args.guide_seq_path)
    cfg.data.target_seq_path = Path(args.target_seq_path)
    cfg.data.metadata_path = Path(args.metadata_path)
    cfg.data.vocab_path = Path(args.vocab_path) if args.vocab_path else None

    cfg.train.device = args.device
    cfg.output_dir = Path(args.output)

    seed_everything(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Data
    full = CRISPRoffTDataset(cfg.data, idxs=None)
    train_idx, val_idx, test_idx = get_splits(len(full.groups), full.groups, seed=cfg.seed)

    vocabs = {}
    if cfg.data.vocab_path is not None:
        with open(cfg.data.vocab_path, "r") as f:
            vocabs = json.load(f)

    train_ds = CRISPRoffTDataset(cfg.data, train_idx, vocabs=vocabs)
    val_ds = CRISPRoffTDataset(cfg.data, val_idx, vocabs=train_ds.vocabs)
    test_ds = CRISPRoffTDataset(cfg.data, test_idx, vocabs=train_ds.vocabs)

    num_scaler = fit_num_scaler(train_ds.X_num)
    collate_fn = get_collate_fn(num_scaler)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn)
    train_eval_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = create_model(cfg, train_ds.vocabs)

    # Train
    trainer = Trainer(cfg, model, train_loader, val_loader)
    trainer.train()

    train_metrics = trainer.metric(train_eval_loader)
    print("\nTrain metrics:", " | ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))
    
    val_metrics = trainer.metric(val_loader)
    print("Val metrics:", " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

    test_metrics = trainer.metric(test_loader)
    print("Test metrics:", " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    trainer.save_predictions(train_loader, cfg.output_dir / "predictions_train.npz")
    trainer.save_predictions(val_loader, cfg.output_dir / "predictions_val.npz")
    trainer.save_predictions(test_loader, cfg.output_dir / "predictions_test.npz")


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
