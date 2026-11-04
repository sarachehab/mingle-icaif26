import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd

from mingle.utils.runner.input import Config
from mingle.utils.runner.output.metadata import RunMetadata
from mingle.utils.runner.output.window import WindowInfo
from mingle.utils.runner.output.constants import (
    CONFIG_FILE, METADATA_FILE, WINDOW_FILE, SUMMARY_FILE, HISTORY_FILE,
    get_window_dir, F_FILE, B_FILE, W_FILE,
    WINDOWS_DIR, WINDOWS_RE,
    IN_RETURNS_FILE, OUT_RETURNS_FILE, WEIGHTS_FILE, RETURN_COL,
)

ExperimentLoaderLogger = logging.getLogger(__name__)

@dataclass
class LoadedAllocation:
    in_returns: pd.Series
    out_returns: pd.Series
    weights: pd.DataFrame

    # @property
    # def returns(self) -> pd.Series:
    #     return self.out_returns

@dataclass
class LoadedWindow:
    info: WindowInfo
    history: pd.DataFrame
    F: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None
    W: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None
    spectral_L: Optional[np.ndarray] = None


class ExperimentLoader:

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def load_config(self) -> Config:
        return Config.from_yaml(self.output_dir / CONFIG_FILE)

    def load_metadata(self) -> RunMetadata:
        with (self.output_dir / METADATA_FILE).open() as f:
            return RunMetadata.model_validate(json.load(f))

    def load_allocation(self) -> LoadedAllocation:
        in_returns = pd.read_csv(self.output_dir / IN_RETURNS_FILE, index_col=0, parse_dates=True).squeeze()
        out_returns = pd.read_csv(self.output_dir / OUT_RETURNS_FILE, index_col=0, parse_dates=True).squeeze()
        weights = pd.read_csv(self.output_dir / WEIGHTS_FILE, index_col=0, parse_dates=True)
        in_returns.name = RETURN_COL
        out_returns.name = RETURN_COL
        return LoadedAllocation(in_returns=in_returns, out_returns=out_returns, weights=weights)

    def load_summary(self) -> pd.DataFrame:
        return pd.read_csv(self.output_dir / SUMMARY_FILE)

    def list_windows(self) -> list[int]:
        windows_dir = self.output_dir / WINDOWS_DIR
        indices = []
        for p in windows_dir.iterdir():
            m = WINDOWS_RE.match(p.name)
            if m and p.is_dir():
                indices.append(int(m.group(1)))
        return sorted(indices)

    def load_window(self, window_idx: int, load_arrays: bool = True) -> LoadedWindow:
        window_dir = self.output_dir / get_window_dir(window_idx)

        with (window_dir / WINDOW_FILE).open() as f:
            info = WindowInfo.model_validate(json.load(f))

        history = pd.read_csv(window_dir / HISTORY_FILE)

        array_path_constructor = lambda x: window_dir / x
        return LoadedWindow(
            info=info,
            history=history,
            F=self._load_arrays(array_path_constructor(F_FILE), load=load_arrays),
            B=self._load_arrays(array_path_constructor(B_FILE), load=load_arrays),
            W=self._load_arrays(array_path_constructor(W_FILE), load=load_arrays),
        )

    def iter_windows(self, load_arrays: bool = True) -> Generator[LoadedWindow, None, None]:
        for idx in self.list_windows():
            yield self.load_window(idx, load_arrays=load_arrays)

    def _load_arrays(self, path: Path, load: bool):

        if not load: return None
        if path.exists(): return np.load(path)

        ExperimentLoaderLogger.info(f"Could not find file {path}. Returning NONE.")
        return None
