from pydantic import BaseModel, Field

import mingle.configs.defaults as defaults
from mingle.utils.rebalancing.frequency import RebalancingFrequency
from mingle.utils.runner.input.base import BaseConfig


class RebalancingConfig(BaseConfig):
    frequency: RebalancingFrequency = defaults.REBALANCING_FREQUENCY
    lookback_horizon: int = Field(default=defaults.LOOKBACK_HORIZON, gt=0)
    ewma_weights: float = Field(default=1.0, gt=0.0, le=1.0)
