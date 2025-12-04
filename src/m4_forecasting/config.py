"""
There are two ways to approach variables.
The first one (used in this repo) uses a dataclass.

The second one uses variables in capital
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable
from typing import Optional

from utilsforecast.losses import mae, smape


@dataclass
class PipelineConfig:
    # Data Params
    group: str = 'Hourly'
    horizon: int = 24
    freq: str = 'h'
    test_size: int = 48  # Hold out last 48h
    n_series_debug: Optional[int] = 0  # Define a subset of ids for debug puproses

    # Optimization Params
    n_windows: int = 5
    n_trials: int = 30
    num_threads: int = 8

    # Evaluation
    metrics: List[Callable] = None

    # Paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    train_model_filename: str = "train_m4_hourly.pkl"
    prod_model_filename: str = "prod_m4_hourly.pkl"

    # Training flag
    skip_training: bool = False

    def __post_init__(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.metrics is None:
            self.metrics = [mae, smape]

    @property
    def training_model_path(self) -> Path:
        return self.model_dir / self.train_model_filename

    @property
    def production_model_path(self) -> Path:
        return self.model_dir / self.prod_model_filename

LGBM_MODEL_NAME: str = 'lgb'
