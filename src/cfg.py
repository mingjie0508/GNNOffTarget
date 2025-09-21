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
    batch_size: int = 128
    device: str = "cuda"
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class ModelConfig:
    seq_in_dim: int = 1536
    num_in_dim: int = 2
    cat_in_dims: dict = {"chr": 8, "strand": 2, "cas9_type": 8, "source": 8}
    metadata_out_dim: int = 32
    hidden_dim: int = 256
    dropout: float = 0.2


@dataclass
class DataConfig:
    guide_seq_path: Path = Path()
    target_seq_path: Path = Path()
    metadata_path: Path = Path()
    vocab_path: Path | None = None


@dataclass
class RunConfig:
    seed: int = 2025
    output_dir: Path = Path()

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
