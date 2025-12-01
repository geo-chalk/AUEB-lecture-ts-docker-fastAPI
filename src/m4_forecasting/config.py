from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable
from utilsforecast.losses import mae, smape
from typing import Optional

@dataclass
class PipelineConfig:
    # Data Params
    group: str = 'Hourly'
    horizon: int = 24
    freq: str = 'h'
    test_size: int = 48  # Hold out last 48h
    n_series_debug: Optional[int] = 20  # Set to None to run full dataset

    # Optimization Params
    n_windows: int = 2
    n_trials: int = 20
    num_threads: int = 8

    # Evaluation
    metrics: List[Callable] = None

    # Paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    model_filename: str = "best_m4_hourly.pkl"

    def __post_init__(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.metrics is None:
            self.metrics = [mae, smape]

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename