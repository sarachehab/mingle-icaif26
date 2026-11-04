from pydantic import Field

import mingle.configs.defaults as defaults
from mingle.utils.runner.input.base import BaseConfig


class AdmmConfig(BaseConfig):
    nb_factors: int = Field(default=defaults.NB_FACTORS, gt=0)
    alpha: float = Field(default=defaults.ALPHA, gt=0)
    beta: float = Field(default=defaults.BETA, ge=0)
    lambda_l: float = Field(default=defaults.LAMBDA_L, ge=0)

    eps: float = Field(default=defaults.EPS, gt=0)
    delta: float = Field(default=defaults.DELTA, ge=0)
    max_epochs: int = Field(default=defaults.MAX_EPOCHS, gt=0)

    abs_eps: float = Field(default=defaults.ABS_EPS, gt=0)
    rel_eps: float = Field(default=defaults.REL_EPS, gt=0)
    abs_eps_B: float = Field(default=defaults.ABS_EPS_B, gt=0)
    rel_eps_B: float = Field(default=defaults.REL_EPS_B, gt=0)

    rho_B: float = Field(default=defaults.RHO_B, gt=0)
    rho_W: float = Field(default=defaults.RHO_W, gt=0)
    rho_d: float = Field(default=defaults.RHO_D, gt=0)
    adapt_every: int = Field(default=defaults.ADAPT_EVERY, gt=0)
    mu: float = Field(default=defaults.MU, gt=0)
    tau: float = Field(default=defaults.TAU, gt=0)
    rho_min: float = Field(default=defaults.RHO_MIN, gt=0)
    rho_max: float = Field(default=defaults.RHO_MAX, gt=0)

    decay: float = Field(default=defaults.DECAY, gt=0, le=1)
    use_low_rank_prior: bool = Field(default=True)
