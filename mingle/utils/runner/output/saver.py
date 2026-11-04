import yaml
import json
import numpy as np
import pandas as pd

from dataclasses import asdict
from pathlib import Path

from mingle.utils.runner.input import Config
from mingle.utils.runner.output.window import WindowInfo
from mingle.utils.runner.output.metadata import DatasetMetadata, RunMetadata
from mingle.utils.runner.output.constants import (
    CONFIG_FILE, METADATA_FILE, WINDOW_FILE, SUMMARY_FILE, HISTORY_FILE,
    get_window_dir, F_FILE, B_FILE, Y_FILE, W_FILE,
    IN_RETURNS_FILE, OUT_RETURNS_FILE, WEIGHTS_FILE, RETURN_COL
)
from mingle.utils.rebalancing.algorithms.admm import AdmmIteration

class ExperimentSaver:
    
    def __init__(self, output_dir: Path, git_hash: str, save_arrays: bool = True):

        self.output_dir = output_dir
        self.save_arrays = save_arrays
        self.git_hash = git_hash

        self._window_records: list[WindowInfo] = []

        self.output_dir.mkdir(parents=True)

    def save_config(self, config: Config) -> None:

        config_dict = config.model_dump(mode="json")

        with (self.output_dir / CONFIG_FILE).open("w") as f:
            yaml.safe_dump(
                config_dict,
                f,
                sort_keys=False,
                default_flow_style=False,
            )

    def save_metadata(self, dataset: DatasetMetadata) -> None:

        metadata = RunMetadata(
            git_hash = self.git_hash,
            dataset = dataset,
        )

        with (self.output_dir / METADATA_FILE).open("w") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2)

    def save_window(self, window_idx: int, iteration: AdmmIteration):

        window_dir = self.output_dir / get_window_dir(window_idx)
        window_dir.mkdir(parents=True, exist_ok=False)

        W = iteration.W
        train_start, train_end = iteration.start_train, iteration.end_train
        test_start, test_end = iteration.start_test, iteration.end_test
        history = iteration.history

        # T comes from whichever representation is present
        rep = iteration.F if iteration.F is not None else iteration.Y
        T = rep.shape[1]

        final = history[-1] if history else None

        window_info = WindowInfo(
            window_idx=window_idx,
            train_start_raw=train_start.date(),
            train_end_raw=train_end.date(),
            test_start_raw=test_start.date(),
            test_end_raw=test_end.date(),
            T=int(T),
            epochs_run=len(history),
            terminates=bool(iteration.terminates),
            duration_seconds=float(iteration.duration_seconds),
            final_rB=getattr(final, 'rB', None) if final else None,
            final_rW=getattr(final, 'rW', 0.0) if final else 0.0,
            final_rd=getattr(final, 'rd', 0.0) if final else 0.0,
            final_sB=getattr(final, 'sB', None) if final else None,
            final_sW=getattr(final, 'sW', 0.0) if final else 0.0,
            final_sd=getattr(final, 'sd', 0.0) if final else 0.0,
        )

        with (window_dir / WINDOW_FILE).open("w") as f:
            json.dump(window_info.model_dump(mode="json"), f, indent=2)

        if self.save_arrays:
            if iteration.F is not None:
                np.save(window_dir / F_FILE, iteration.F)
                np.save(window_dir / B_FILE, iteration.B)
            if iteration.Y is not None:
                np.save(window_dir / Y_FILE, iteration.Y)
            np.save(window_dir / W_FILE, W)

        history_df = self._history_to_dataframe(history)
        history_df.to_csv(window_dir / HISTORY_FILE, index=False)

        self._window_records.append(window_info)

    def save_summary(self) -> None:
        summary = pd.DataFrame(
            [record.model_dump(mode="json") for record in self._window_records]
        )
        summary.to_csv(self.output_dir / SUMMARY_FILE, index=False)

    def save_performance(
        self,
        in_ret: pd.Series,
        out_ret: pd.Series,
        weights: pd.DataFrame,
    ) -> None:
        in_ret = in_ret.rename(RETURN_COL)
        out_ret = out_ret.rename(RETURN_COL)
        in_ret.to_csv(self.output_dir / IN_RETURNS_FILE)
        out_ret.to_csv(self.output_dir / OUT_RETURNS_FILE)
        weights.to_csv(self.output_dir / WEIGHTS_FILE)
    
    @staticmethod
    def _history_to_dataframe(history: list) -> pd.DataFrame:
        return pd.DataFrame(
            [asdict(metric) for metric in history]
        )
