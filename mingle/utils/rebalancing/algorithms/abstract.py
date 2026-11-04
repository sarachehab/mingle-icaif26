from abc import ABC
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RebalanceDebug:
    terminates: bool = True
    start_train: pd.Timestamp = None
    end_train: pd.Timestamp = None
    start_test: pd.Timestamp = None
    end_test: pd.Timestamp = None
    duration_seconds: float = 0.0

    def set_dates(self, train_dates, test_dates):
        self.start_train = train_dates.min()
        self.end_train   = train_dates.max()
        self.start_test  = test_dates.min()
        self.end_test    = test_dates.max()

class ModelWrapper(ABC):

    def __init__(self, noise_scale: float = 0.0, seed: int = 0):
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)

    def get_weights(
        self,
        X_train: np.ndarray,
        asset_names = None,
    ) -> tuple[np.ndarray, RebalanceDebug]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support get_weights."
        )

    def backtest(
        self,
        prices_df: np.ndarray,
        data: RebalanceDebug,
    ) -> np.ndarray:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support backtesting."
        )

    def add_noise(self, X_train):
        if self.noise_scale > 0:
            return X_train + self.rng.normal(0, self.noise_scale * X_train.std(), X_train.shape)
        return X_train
