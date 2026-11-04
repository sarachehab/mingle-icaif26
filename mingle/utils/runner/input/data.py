import pandas as pd

from pathlib import Path
from pydantic import Field
from datetime import date
from typing import Optional

import mingle.configs.defaults as defaults
from mingle.utils.enums import Index, BloombergField
from mingle.configs.constants import DATA_DIR
from mingle.utils.runner.input.base import BaseConfig


class DataConfig(BaseConfig):
    indices: list[Index] = Field(
        default_factory=lambda:defaults.INDICES
    )
    data_dir: Path = DATA_DIR
    start_raw: date = Field(default=defaults.START_DATE.date())
    end_raw: date = Field(default=defaults.END_DATE.date())
    top_n_by_volume: int = Field(default=defaults.TOP_N_BY_VOLUME, gt=0)
    price_field: BloombergField = defaults.PRICE_FIELD

    # If set, prices are loaded through CustomDataFetcher from this path, and
    # indices, data_dir, top_n_by_volume and price_field are ignored.
    custom_path: Optional[Path] = None

    @property
    def start_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.start_raw)

    @property
    def end_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.end_raw)
