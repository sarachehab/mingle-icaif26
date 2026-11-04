import logging
import time
import pandas as pd

from mingle.utils.rebalancing.frequency import RebalancingFrequency
from mingle.utils.rebalancing.algorithms import ModelWrapper, RebalanceDebug
from mingle.utils.transforms import log_returns

RebalancingLogger = logging.getLogger(__name__)

class PortfolioRebalancing:

    def __init__(self, freq: RebalancingFrequency, window_months: int = 1, ewma_alpha: float = 1.0):
        self.freq = freq
        self.window_months = window_months
        self.ewma_alpha = ewma_alpha
    
    def date_pace_generator(self, dates):
        start_date = dates.min()
        end_date   = dates.max()

        get_window = lambda interval: dates[
            (dates >= interval.left) & (dates < interval.right)
        ]

        date_intervals = pd.interval_range(
            start=start_date, end=end_date,
            freq=self.freq.value, closed="left"
        )

        for i, test_dates_interval in enumerate(date_intervals[1:], start=1):
            
            # Skip until we have a full training window
            if i < self.window_months:
                RebalancingLogger.debug(
                    f"Skipping rebalance {i}, as window is not long enough"
                    f"({i}/{self.window_months} months)"
                )
                continue
             
            # Use month_window monthly windows
            train_start_idx = max(0, i - self.window_months)
            train_intervals = date_intervals[train_start_idx:i]

            train_dates = dates[
                (dates >= train_intervals[0].left) &
                (dates < train_intervals[-1].right)
            ]
            test_dates = get_window(test_dates_interval)

            # Skip incomplete window
            if len(train_dates) == 0 or len(test_dates) == 0:
                RebalancingLogger.warning(
                    f"Skipping incomplete window at rebalance {i}"
                )
                continue

            yield train_dates, test_dates

    def run(self, prices_df: pd.DataFrame, model_wrapper: ModelWrapper):

        debug_info: list[RebalanceDebug] = []

        dates = prices_df.index
        out_of_sample_portfolio_returns = pd.Series(index=dates, dtype=float)
        in_sample_portfolio_returns = pd.Series(index=dates, dtype=float)
        asset_names = prices_df.columns
        weights_df = pd.DataFrame(
            index=pd.Index([], name="Rebalance date"),
            columns=asset_names,
            dtype=float,
        )

        for rebalancing_count, (train_dates, test_dates) in enumerate(self.date_pace_generator(dates)):

            RebalancingLogger.info(f"Running allocation {rebalancing_count}: {train_dates.min()} to {train_dates.max()}")

            # Preprocessing data
            X_train_log = self._preprocess_train(prices_df, train_dates)
            X_train_dates, X_train_simple = self._preprocess_test(prices_df, train_dates)
            X_test_dates, X_test = self._preprocess_test(prices_df, test_dates)

            # Run porfolio clustering algorithm
            t0 = time.perf_counter()
            portfolio_weights, debug = model_wrapper.get_weights(X_train=X_train_log, asset_names=asset_names)
            debug.duration_seconds = time.perf_counter() - t0

            # Exponential weights smoothing
            if not weights_df.empty: portfolio_weights = self.ewma_alpha * portfolio_weights + (1 - self.ewma_alpha) * weights_df.iloc[-1].values

            # Save portfolio out-of-sample price point and weights
            test_portfolio_simple_return = X_test.T @ portfolio_weights

            RebalancingLogger.info(f"Evaluating on {X_test_dates.min()} to {X_test_dates.max()}")
            out_of_sample_portfolio_returns.loc[X_test_dates] = test_portfolio_simple_return
            weights_df.loc[X_test_dates.min()] = portfolio_weights

            # Save portfolio performance in-sample, only for first occurence of dates  
            unseen = X_train_dates[in_sample_portfolio_returns.loc[X_train_dates].isna()]
            unseen_mask = pd.Index(X_train_dates).isin(unseen)
            train_portfolio_simple_return = X_train_simple.T @ portfolio_weights
            in_sample_portfolio_returns.loc[unseen] = train_portfolio_simple_return[unseen_mask]

            # Save debug information
            debug.set_dates(train_dates, test_dates)
            debug_info.append(debug)

        return in_sample_portfolio_returns.dropna(), out_of_sample_portfolio_returns.dropna(), weights_df, debug_info

    def backtest(self, prices_df: pd.DataFrame, model_wrapper: ModelWrapper, per_rebalance_data: list[RebalanceDebug]):

        dates = prices_df.index
        asset_names = prices_df.columns

        out_of_sample_portfolio_returns = pd.Series(index=dates, dtype=float)
        in_sample_portfolio_returns = pd.Series(index=dates, dtype=float)
        weights_df = pd.DataFrame(
            index=pd.Index([], name="Rebalance date"),
            columns=asset_names,
            dtype=float,
        )

        for rebalancing_count, data in enumerate(per_rebalance_data):

            RebalancingLogger.info(f"Backtesting allocation {rebalancing_count}: {data.start_train} to {data.end_train}")

            # Extract train/test date masks
            train_dates = dates[(dates >= data.start_train) & (dates <= data.end_train)]
            test_dates  = dates[(dates >= data.start_test)  & (dates <= data.end_test)]

            # Preprocess
            X_train_log = self._preprocess_train(prices_df, train_dates)
            X_train_dates, X_train_simple = self._preprocess_test(prices_df, train_dates)
            X_test_dates, X_test = self._preprocess_test(prices_df, test_dates)

            # Delegate to model_wrapper backtest
            portfolio_weights = model_wrapper.backtest(X_train_log, data)

            # Exponential weights smoothing
            if not weights_df.empty: portfolio_weights = self.ewma_alpha * portfolio_weights + (1 - self.ewma_alpha) * weights_df.iloc[-1].values

            # Evaluate on test set
            test_portfolio_simple_return = X_test.T @ portfolio_weights
            out_of_sample_portfolio_returns.loc[X_test_dates] = test_portfolio_simple_return          
            weights_df.loc[X_test_dates.min()] = portfolio_weights

            # Save portfolio performance in-sample, only for first occurence of dates  
            unseen = X_train_dates[in_sample_portfolio_returns.loc[X_train_dates].isna()]
            unseen_mask = pd.Index(X_train_dates).isin(unseen)
            train_portfolio_simple_return = X_train_simple.T @ portfolio_weights
            in_sample_portfolio_returns.loc[unseen] = train_portfolio_simple_return[unseen_mask]

        return in_sample_portfolio_returns.dropna(), out_of_sample_portfolio_returns.dropna(), weights_df

    def _preprocess_train(self, prices_df, train_dates):
        return log_returns(prices_df, train_dates.min(), train_dates.max())

    def _preprocess_test(self, prices_df, test_dates):
        X_test_df = prices_df.loc[test_dates].pct_change().dropna()
        return X_test_df.index, X_test_df.to_numpy().T
