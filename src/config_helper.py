# config_schema.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    # General
    experiment_name: str = "debug"
    seed: int = 42

    # Data
    data_dir: str = "./data"
    num_workers: int = 4

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 10

    # Model
    model_name: str = "resnet18"
    hidden_dim: int = 256

    # Logging
    output_dir: str = "./outputs"
    log_every: int = 50
