import pandas as pd

from datetime import date
from pydantic import BaseModel
from typing import Optional

class WindowInfo(BaseModel):
    window_idx: int

    train_start_raw: date
    train_end_raw: date
    test_start_raw: date
    test_end_raw: date
    T: int

    nb_clusters: Optional[int] = 0
    epochs_run: int
    terminates: bool
    duration_seconds: Optional[float] = None

    final_rB: Optional[float] = None
    final_rW: float = 0.0
    final_rd: float = 0.0
    final_sB: Optional[float] = None
    final_sW: float = 0.0
    final_sd: float = 0.0

    @property
    def train_start(self) -> pd.Timestamp:
        return pd.to_datetime(self.train_start_raw, format="%Y-%m-%d")

    @property
    def train_end(self) -> pd.Timestamp:
        return pd.to_datetime(self.train_end_raw, format="%Y-%m-%d")
    
    @property
    def test_start(self) -> pd.Timestamp:
        return pd.to_datetime(self.test_start_raw, format="%Y-%m-%d")
    
    @property
    def test_end(self) -> pd.Timestamp:
        return pd.to_datetime(self.test_end_raw, format="%Y-%m-%d")
