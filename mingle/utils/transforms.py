import numpy as np
import pandas as pd


def log_returns(prices_df: pd.DataFrame, start=None, end=None) -> np.ndarray:
    """
    Compute log-returns from a price DataFrame and return as (n_assets, T).

    Args:
        prices_df : Price DataFrame indexed by date, columns = tickers.
        start, end : Inclusive dates

    Returns:
        Slice of log returns
    """
    subset = prices_df.loc[start:end] if (start is not None or end is not None) else prices_df
    return np.log(subset).diff().dropna().to_numpy().T
