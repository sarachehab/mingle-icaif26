from typing import Literal, Annotated, Union
from pydantic import TypeAdapter, Tag, Discriminator

import mingle.configs.defaults as defaults
from mingle.utils.enums import AllocationStrategy, CutType
from mingle.utils.runner.input.base import StrategyConfig, strategy_discriminator


class BenchmarkConfig(StrategyConfig):

    @classmethod
    def _adapter(cls) -> TypeAdapter:
        return BenchmarkConfigAdapter


class HrpConfig(BenchmarkConfig):
    type: Literal[AllocationStrategy.HRP] = AllocationStrategy.HRP
    decay: float = 1.0
    linkage_method: Literal["single", "ward", "complete", "average"] = "single"
    ridge: float = 1e-8


class MvoConfig(BenchmarkConfig):
    type: Literal[AllocationStrategy.MVO] = AllocationStrategy.MVO
    decay: float = 1.0
    ridge: float = 1e-8


class EwConfig(BenchmarkConfig):
    type: Literal[AllocationStrategy.EW] = AllocationStrategy.EW


class CutConfig(BenchmarkConfig):
    type: Literal[AllocationStrategy.CUT] = AllocationStrategy.CUT
    decay: float = 1.0
    cut_type: CutType = defaults.CUT_TYPE
    splits_count: int = defaults.CUT_V_COUNT


class FactorConfig(BenchmarkConfig):
    type: Literal[AllocationStrategy.FACTOR] = AllocationStrategy.FACTOR
    nb_factors: int = 6
    decay: float = 1.0
    cut_type: CutType = defaults.CUT_TYPE
    splits_count: int = defaults.CUT_V_COUNT


BenchmarkConfigUnion = Annotated[
    Union[
        Annotated[MvoConfig, Tag(AllocationStrategy.MVO.name)],
        Annotated[EwConfig, Tag(AllocationStrategy.EW.name)],
        Annotated[HrpConfig, Tag(AllocationStrategy.HRP.name)],
        Annotated[CutConfig, Tag(AllocationStrategy.CUT.name)],
        Annotated[FactorConfig, Tag(AllocationStrategy.FACTOR.name)],
    ],
    Discriminator(strategy_discriminator),
]


BenchmarkConfigAdapter = TypeAdapter(BenchmarkConfigUnion)
