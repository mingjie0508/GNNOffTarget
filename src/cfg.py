from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

@dataclass
class TrainConfig:
    epochs: int = 10
    device: str = "cuda"
    optim: OptimConfig = field(default_factory=OptimConfig)

@dataclass
class ModelConfig:
    name: str = "gcn"           # "gcn" or "gat"
    in_channels: int = 8
    hidden_channels: int = 16   # TODO change this
    out_channels: int = 16
    dropout: float = 0.0

    # only used by GAT
    heads: int = 4

@dataclass
class DataConfig:
    root: Path = Path("data")
    batch_size: int = 32
    num_workers: int = 4

@dataclass
class PathsConfig:
    out_dir: Path = Path("outputs/")

@dataclass
class RunConfig:
    seed: int = 2025
    model_name: str = "GCN"
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)