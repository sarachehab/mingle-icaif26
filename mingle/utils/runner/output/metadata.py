import platform
import sys

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    prices_shape: tuple[int, int]
    n_assets: int
    asset_names: list[str]
    n_dates: int
    missing_fraction: float | None = None

class RunMetadata(BaseModel):
    git_hash: str | None = None
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    python_version: str = sys.version
    platform: str = platform.platform()
    dataset: Optional[DatasetMetadata] = None
