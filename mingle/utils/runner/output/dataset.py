from pydantic import BaseModel

class DatasetMetadata(BaseModel):
    prices_shape: tuple[int, int]
    n_assets: int
    n_dates: int
    missing_fraction: float | None = None