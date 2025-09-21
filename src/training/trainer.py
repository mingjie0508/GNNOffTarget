from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import pearsonr
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..builders import create_early_stopper, create_loss_fn, create_optimizer
from ..cfg import RunConfig
from .earlystop import EarlyStopper


class FitState:
    def __init__(self):
        self.epoch: int = 0
        self.global_step: int = 0
        self.metrics: dict[str, float] = {}
        self.best_metric: Optional[dict[str, float]] = None
        self.best_epoch: Optional[int] = None


class Trainer:
    def __init__(
        self,
        cfg: RunConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        criterion: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        early_stopper: EarlyStopper | None = None,
    ):
        self.cfg = cfg
        self.device = cfg.train.device
        self.out_dir = cfg.output_dir

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or create_loss_fn()
        self.optimizer = optimizer or create_optimizer(self.model, cfg.train.optim)
        self.early_stopper = early_stopper or create_early_stopper(cfg.train.early_stop)

        self.best_state_dict = None
        self.state = FitState()

    def train(self) -> FitState:
        for epoch in range(self.cfg.train.epochs):
            self.state.epoch = epoch
            train_loss, train_mae, train_pcc = self._run_one_epoch(self.train_loader, train=True)
            val_loss, val_mae, val_pcc = self._run_one_epoch(self.val_loader, train=False)

            metrics = {
                "train_loss": train_loss,
                "train_mae": train_mae,
                "train_pcc": train_pcc,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_pcc": val_pcc,
            }

            if self.state.best_metric is None or metrics["val_loss"] < self.state.best_metric["val_loss"]:
                self.state.best_metric = metrics.copy()
                self.state.best_epoch = epoch
                self._snapshot_best_weights()

            if self.early_stopper.step(metrics["val_loss"]):
                print(f"Early stopping at epoch {epoch}")
                break

            self._print_epoch_line(epoch, metrics)

        self._restore_best_weights()

        return self.state

    def _run_one_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float, float]:
        """Run one epoch and return average metrics."""

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, total_mae = 0.0, 0.0
        n = 0

        y_probs = []
        y_trues = []

        for batch in loader:
            _, x_seq, x_num, x_chr, x_strand, x_cas9, x_source, y = batch

            x_seq = x_seq.to(self.device)
            x_num = x_num.to(self.device)
            x_chr = x_chr.to(self.device)
            x_strand = x_strand.to(self.device)
            x_cas9 = x_cas9.to(self.device)
            x_source = x_source.to(self.device)
            y = y.to(self.device)

            if train:
                self.optimizer.zero_grad()

            preds = self.model(x_seq, x_num, x_chr, x_strand, x_cas9, x_source)
            loss = self.criterion(preds, y)
            probs = torch.sigmoid(preds.detach())

            if train:
                loss.backward()
                self.optimizer.step()
                self.state.global_step += 1

            total_loss += loss.detach().item() * y.size(0)
            total_mae += (probs.detach() - y).abs().sum().item()
            n += y.size(0)

            y_probs.append(probs.cpu())
            y_trues.append(y.detach().cpu())

        avg_loss = total_loss / n
        mae = total_mae / n

        y_probs = torch.cat(y_probs).numpy()
        y_trues = torch.cat(y_trues).numpy()
        pcc = pearsonr(y_probs, y_trues)[0]

        return avg_loss, mae, pcc

    def _print_epoch_line(self, epoch: int, metrics: dict[str, float]) -> None:
        if metrics:
            line = f"[Epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())

        else:
            line = f"[Epoch {epoch}] (no metrics)"

        print(line)

    def _snapshot_best_weights(self) -> None:
        self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        torch.save(self.best_state_dict, self.out_dir / "best_model.pt")

    def _restore_best_weights(self) -> None:
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

    def load_state_dict(self, state, strict: bool = True) -> None:
        """
        Load model weights from a .pt checkpoint path.

        Simple Usage:
        state = torch.load(path, map_location=self.device)
        """
        self.model.load_state_dict(state, strict=strict)
        self.model.to(self.device).eval()

    def predict(self, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (row_id, y_pred, y_true)."""

        self.model.eval()
        row_ids, y_probs, y_trues = [], [], []

        for batch in loader:
            row_id, x_seq, x_num, x_chr, x_strand, x_cas9, x_source, y = batch

            x_seq = x_seq.to(self.device)
            x_num = x_num.to(self.device)
            x_chr = x_chr.to(self.device)
            x_strand = x_strand.to(self.device)
            x_cas9 = x_cas9.to(self.device)
            x_source = x_source.to(self.device)
            y = y.to(self.device)

            pred = self.model(x_seq, x_num, x_chr, x_strand, x_cas9, x_source)
            prob = torch.sigmoid(pred.detach())

            row_ids.extend(row_id)
            y_probs.append(prob.cpu())
            y_trues.append(y.cpu())

        row_ids = np.array(row_ids, dtype=np.int64)
        y_probs = torch.cat(y_probs).numpy()
        y_trues = torch.cat(y_trues).numpy()
        return row_ids, y_probs, y_trues

    @torch.no_grad()
    def metric(self, loader) -> dict[str, float]:
        _, y_prob, y_true = self.predict(loader)

        mse = float(np.mean((y_prob - y_true) ** 2))
        mae = float(np.mean(np.abs(y_prob - y_true)))
        pcc = float(pearsonr(y_true, y_prob)[0])

        metrics = {"loss": mse, "mae": mae, "pcc": pcc}
        return metrics

    @torch.no_grad()
    def save_predictions(self, loader: DataLoader, out_path: Path) -> None:
        """Save predictions and true labels to a compressed .npz file."""
        row_ids, y_prob, y_true = self.predict(loader)
        np.savez_compressed(out_path, row_ids=row_ids, y_prob=y_prob, y_true=y_true)
