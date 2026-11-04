from pydantic import Field
from mingle.utils.runner.input.base import BaseConfig

import mingle.configs.defaults as defaults

class ExperimentConfig(BaseConfig):
    name: str = "base"
    description: str = "Baseline ADMM representation learning"
    seed: int = Field(default=defaults.SEED, ge=0)
    noise_scale: float = Field(default=0.0, ge=0)
