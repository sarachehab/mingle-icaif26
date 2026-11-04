import numpy as np
import pandas as pd

from abc import abstractmethod
from pathlib import Path

from mingle.utils.runner.input import Config
from mingle.utils.runner.output import ExperimentSaver
from mingle.utils.rebalancing import PortfolioRebalancing, AdmmRebalancingWrapper
from mingle.utils.fetcher.bloomberg import BloombergDataFetcher
from mingle.utils.fetcher.custom import CustomDataFetcher
from mingle.utils.runner.output.metadata import DatasetMetadata
from mingle.utils.runner.input.data import DataConfig


def load_prices(data_config: DataConfig) -> pd.DataFrame:
    if data_config.custom_path is not None:
        prices = CustomDataFetcher.get_prices(filepath=data_config.custom_path)
        return prices.loc[data_config.start_date:data_config.end_date]

    price_matrices = [
        BloombergDataFetcher.get_top_volume_price_matrix(
            filepath=data_config.data_dir / index.value / "daily.csv",
            price_field=data_config.price_field,
            date_range=(data_config.start_date, data_config.end_date),
            top_n_by_volume=data_config.top_n_by_volume,
        )
        for index in data_config.indices
    ]

    return pd.concat(
        price_matrices,
        axis=1,
        join="inner",
    )


class BaseRunner:

    def __init__(
        self,
        config: Config,
        output_root: Path,
        git_hash: str,
        save_window: bool,
        save_arrays: bool,
        save_perf: bool,
    ):
        self.config = config
        self.output_root = Path(output_root)
        self.git_hash = git_hash
        self.save_window = save_window
        self.save_arrays = save_arrays
        self.save_perf = save_perf

    @abstractmethod
    def build_model(self) -> AdmmRebalancingWrapper:
        pass

    def run(self) -> None:
        np.random.seed(self.config.experiment.seed)

        # Save run information
        saver = self.build_saver()
        saver.save_config(config=self.config)

        # Load data
        prices_df = self.load_data()
        dataset_metadata = self.build_dataset_metadata(prices_df)
        saver.save_metadata(dataset=dataset_metadata)

        model = self.build_model()
        rebalancing = self.build_rebalancing()

        # Run rebalancing
        in_ret, out_ret, portfolio_weights, debug = rebalancing.run(prices_df=prices_df, model_wrapper=model)

        # Save run outputs
        if self.save_window:
            for window_idx, window in enumerate(debug):
                saver.save_window(window_idx, window)

            saver.save_summary()

        # Save performance
        if self.save_perf:
            saver.save_performance(in_ret=in_ret, out_ret=out_ret, weights=portfolio_weights)

    def load_data(self) -> pd.DataFrame:
        return load_prices(self.config.data)

    def build_rebalancing(self) -> PortfolioRebalancing:
        rebalancing_config = self.config.rebalancing 
        return PortfolioRebalancing(
            freq=rebalancing_config.frequency,
            window_months=rebalancing_config.lookback_horizon,
        )

    def build_saver(self) -> ExperimentSaver:
        return ExperimentSaver(
            output_dir=self.output_root / self.config.experiment.name,
            save_arrays=self.save_arrays,
            git_hash=self.git_hash
        )

    def build_dataset_metadata(self, prices_df) -> DatasetMetadata:
        n_dates, n_assets = prices_df.shape
        missing_fraction = prices_df.isna().sum().sum() / prices_df.size
        return DatasetMetadata(
            prices_shape = prices_df.shape,
            n_assets = n_assets,
            n_dates = n_dates,
            asset_names = prices_df.columns.to_list(),
            missing_fraction = float(missing_fraction),
        )
