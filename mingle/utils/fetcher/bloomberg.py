import pandas as pd
import logging

from pathlib import Path
from typing import Tuple, Optional
from mingle.utils.enums import BloombergField
from mingle.configs.constants import VOLUME_COLUMN

BloombergDataFetcherLogger = logging.getLogger(__name__)

class BloombergDataFetcher:

    @classmethod
    def get_prices(
        cls,
        filepath: Path,
        header: Tuple[int] = (0, 1),
        dateformat: str = "%Y-%m-%d",
        ticker_col: str = "Ticker",
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        field: Optional[BloombergField] = None,
        vol_field: str = VOLUME_COLUMN
    ):
        # Read CSV
        prices = pd.read_csv(
            filepath,
            header=list(header),
        )
        prices.columns = prices.columns.set_names([ticker_col, "Field"])

        # Parse dates
        dates_col = prices.columns[0]
        dates = prices.iloc[:,0]
        parsed_dates = pd.to_datetime(dates, format=dateformat, errors="coerce")
        prices.iloc[:,0] = parsed_dates

        # Drop rows with unparsed dates
        rows_errors_parsing = parsed_dates.isna()
        rows_index_errors_parsing = rows_errors_parsing[rows_errors_parsing].index.values
        prices = prices.dropna(subset=[dates_col])

        BloombergDataFetcherLogger.warning(f"Dropping {len(rows_index_errors_parsing)} rows due to date parsing errors.")
        BloombergDataFetcherLogger.debug(f"First 5 dropped indices are {rows_index_errors_parsing[:5]}")

        # Set date as index
        prices[dates_col] = pd.to_datetime(prices[dates_col])
        prices.set_index(dates_col, inplace=True)
        prices.index.name = "Dates"

        # Force all columns to numbers
        prices = prices.apply(pd.to_numeric, errors='coerce')

        # Filter by field and date range
        if field is not None: prices = prices.loc[:, prices.columns.get_level_values("Field").isin([field.value, vol_field])]
        if date_range is not None: prices = prices.loc[date_range[0]:date_range[1]]

        return prices
    
    @classmethod
    def get_top_volume_price_matrix(
        cls,
        filepath: Path,
        price_field: BloombergField,
        top_n_by_volume: int,
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        volume_field: str = VOLUME_COLUMN,
        drop_assets_with_missing_prices: bool = True,
    ) -> pd.DataFrame:
        raw = cls.get_prices(
            filepath=filepath,
            field=price_field,
            date_range=date_range,
            vol_field=volume_field,
        )

        if drop_assets_with_missing_prices: raw = raw.dropna(axis=1)

        volume = raw.xs(volume_field, level="Field", axis=1)
        avg_volume = volume.mean(axis=0)
        top_assets = avg_volume.nlargest(top_n_by_volume).index

        prices = raw.xs(price_field.value, level="Field", axis=1)
        prices = prices.loc[:, top_assets]

        return prices

    @classmethod
    def get_metadata(
        self,
        filepath: Path,
        ticker_col: str = "Ticker"
    ):
        metadata = pd.read_csv(filepath)
        metadata.columns = metadata.columns.str.lower()
        metadata = metadata.set_index(ticker_col.lower()).sort_index()
        return metadata

    @classmethod
    def get_risk_free_rate(
        self,
        filepath: Path,
        dates,
        column_name: str = "DTB3",
        date_col: str = "observation_date",
        dateformat: str = "%Y-%m-%d",
    ):
        rf = pd.read_csv(filepath)

        # Parse dates
        rf[date_col] = pd.to_datetime(rf[date_col], format=dateformat, errors="coerce")
        rf = rf.dropna(subset=[date_col])

        # Align to requested dates
        rf = rf.set_index(date_col).sort_index()
        rf[column_name] = pd.to_numeric(rf[column_name], errors="coerce")
        rf = rf[column_name].reindex(dates)

        return rf
