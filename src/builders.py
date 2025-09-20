from __future__ import annotations
from typing import Optional, Tuple, Any
import torch
from torch import nn
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from .cfg import RunConfig, OptimConfig
from .gnn import GCNLinkPredictor, GATLinkPredictor


def create_model(cfg: RunConfig) -> nn.Module:
    m = cfg.model

    if m.name.lower() == "gcn":
        return GCNLinkPredictor(
            m.in_channels,
            m.hidden_channels,
            m.out_channels,
            m.dropout,
        )
    elif m.name.lower() == "gat":
        return GATLinkPredictor(
            m.in_channels,
            m.hidden_channels,
            m.out_channels,
            heads=m.heads,
            dropout=m.dropout,
        )
    else:
        raise NotImplementedError(f"Unknown model: {m.name}")


def create_optimizer(model: nn.Module, cfg: OptimConfig) -> Optimizer:
    return AdamW(
        params=model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
