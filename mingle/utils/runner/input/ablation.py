from pydantic import BaseModel
from typing import Any


class AblationConfig(BaseModel):
    name: str
    overrides: list[dict[str, Any]]
