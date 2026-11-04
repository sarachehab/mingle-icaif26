import pandas as pd
import logging

from pathlib import Path

CustomDataFetcherLogger = logging.getLogger(__name__)

class CustomDataFetcher:

    @classmethod
    def get_prices(
        cls,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load the price matrix of the asset universe.

        This is the only input of the representation learning and of the
        rebalancing, which derives log-returns (for training) and simple
        returns (for evaluation) from it.

        Returns:
            prices: pd.DataFrame where
                - the index is a pd.DatetimeIndex of trading days, sorted ascending;
                - there is one column per asset, named by its ticker;
                - values are strictly positive close prices, in a common currency
                  across assets (USD in the paper);
                - there are no missing values: drop assets with gaps, rather than
                  filling them, as the Bloomberg loader does;
        """
        raise NotImplementedError(f"{cls.__name__}.get_prices must be implemented to load custom data.")

    @classmethod
    def get_metadata(
        cls,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load static information about each asset.

        Only used to order assets by sector in Figure 3.

        Returns:
            metadata: pd.DataFrame where
                - the index is named "ticker" and matches the columns of get_prices;
                - there is at least a "gics_sector_name" column; any other
                  columns are ignored.
        """
        raise NotImplementedError(f"{cls.__name__}.get_metadata must be implemented to load custom data.")
