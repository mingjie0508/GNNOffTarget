import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from models.fcnn import FCNNRegressor

from .cfg import EarlyStopperConfig, OptimConfig, RunConfig
from .training.earlystop import EarlyStopper


def create_model(cfg: RunConfig, vocabs: dict) -> nn.Module:
    return FCNNRegressor(
        vocabs=vocabs,
        seq_in_dim=cfg.model.seq_in_dim,
        num_in_dim=cfg.model.num_in_dim,
        cat_in_dims=cfg.model.cat_in_dims,
        metadata_out_dim=cfg.model.metadata_out_dim,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    )


def create_optimizer(model: nn.Module, cfg: OptimConfig) -> Optimizer:
    return AdamW(
        params=model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    
def create_loss_fn():
    return torch.nn.BCEWithLogitsLoss()

def create_early_stopper(cfg: EarlyStopperConfig):
    return EarlyStopper(cfg.patience, cfg.min_delta)
    
