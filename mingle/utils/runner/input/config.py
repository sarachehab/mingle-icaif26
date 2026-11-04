from __future__ import annotations

import yaml

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional

from mingle.utils.runner.input.experiments import ExperimentConfig
from mingle.utils.runner.input.data import DataConfig
from mingle.utils.runner.input.rebalancing import RebalancingConfig
from mingle.utils.runner.input.admm import AdmmConfig
from mingle.utils.runner.input.benchmark import BenchmarkConfigUnion
from mingle.utils.runner.input.allocation import AllocationConfigUnion


class Config(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    rebalancing: RebalancingConfig = Field(default_factory=RebalancingConfig)
    admm: AdmmConfig = Field(default_factory=AdmmConfig)
    benchmark: Optional[BenchmarkConfigUnion] = None
    allocation: Optional[AllocationConfigUnion] = None

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(raw)
