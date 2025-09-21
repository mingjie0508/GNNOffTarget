from pathlib import Path
from typing import Optional
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
        self.metrics: dict[str, float] = {}
        self.best_metric: Optional[dict[str, float]] = None
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
        optimizer: Optimizer,
    ):
        self.cfg = cfg
        self.device = cfg.train.device
        self.out_dir = cfg.output_dir
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
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

    def _run_one_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        """Run one epoch and return average metrics."""

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, total_mse = 0.0, 0.0
        num_batches = 0
        optim = self.optimizer
        criterion = self.criterion

        for batch in loader:
            x_seq, x_num, x_chr, x_strand, x_cas9, x_source, y = (b.to(self.device) for b in batch)
            

            if train:
                optim.zero_grad()

            with torch.cuda.amp.autocast(enabled=(scaler is not None and scaler.is_enabled())):
                preds = self.model(x_seq, x_num, x_chr, x_strand, x_cas9, x_source)
                loss = criterion(preds, y)

            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()
                self.state.global_step += 1

            total_loss += loss.detach().item() * y.size(0)
            total_mae  += (preds.detach() - y).abs().sum().item()
            n += y.size(0)

        avg_loss = total_loss / max(n, 1)
        avg_mae  = total_mae  / max(n, 1)
        return avg_loss, avg_mae

    def _train_one_epoch(self, epoch: int) -> tuple[float, float]:
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

    def _validate_one_epoch(self, epoch: int) -> tuple[float, float]:
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

    def _print_epoch_line(self, epoch: int, metrics: dict[str, float]) -> None:
        if metrics:
            line = f"[Epoch {epoch}] " + " ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
            )

        else:
            line = f"[Epoch {epoch}] (no metrics)"

        print(line)
