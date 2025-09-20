from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .cfg import RunConfig
from .builders import create_optimizer


class FitState:
    def __init__(self):
        self.epoch: int = 0
        self.global_step: int = 0
        self.metrics: Dict[str, float] = {}
        self.best_metric: Optional[Dict[str, float]] = None
        self.best_epoch: Optional[int] = None


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.bad_epochs: int = 0

    def step(self, current: float) -> bool:
        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class Trainer:
    def __init__(
        self,
        cfg: RunConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        criterion: nn.Module,
        optimizer: Optional[Optimizer] = None,
        out_dir: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.model = model.to(cfg.train.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.out_dir = out_dir or cfg.paths.out_dir
        self.optimizer = optimizer or create_optimizer(self.model, cfg.train.optim)
        self.state = FitState()

    def train(self) -> FitState:
        for epoch in range(self.cfg.train.epochs):
            self.state.epoch = epoch
            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._validate_one_epoch(epoch)

            if (
                self.state.best_metric is None
                or val_metrics["val_loss"] < self.state.best_metric["val_loss"]
            ):
                self.state.best_metric = val_metrics.copy()
                self.state.best_epoch = epoch

            if self.early_stopper:
                if self.early_stopper.step(val_metrics["val_loss"]):
                    print(f"Early stopping at epoch {epoch}")
                    break

            self._print_epoch_line(epoch, {**train_metrics, **val_metrics})

        return self.state

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch and return average metrics."""

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.cfg.train.device)
            y = y.to(self.cfg.train.device)

            # forward
            logits = self.model(x)
            loss = self.criterion(logits, y)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # logging
            self.state.global_step += 1
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def _validate_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch and return average metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_loader):
                x = x.to(self.cfg.train.device)
                y = y.to(self.cfg.train.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_loss / num_batches
        return {"val_loss": avg_val_loss}

    def _print_epoch_line(self, epoch: int, metrics: Dict[str, float]) -> None:
        if metrics:
            line = f"[Epoch {epoch}] " + " ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
            )

        else:
            line = f"[Epoch {epoch}] (no metrics)"

        print(line)
