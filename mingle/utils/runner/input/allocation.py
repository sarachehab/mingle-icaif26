from typing import Literal, Annotated, Union
from pydantic import TypeAdapter, Tag, Discriminator

import mingle.configs.defaults as defaults
from mingle.utils.enums import CutType, AllocationStrategy
from mingle.utils.runner.input.base import StrategyConfig, strategy_discriminator


class AllocationConfig(StrategyConfig):

    @classmethod
    def _adapter(cls) -> TypeAdapter:
        return AllocationConfigAdapter


class CutConfig(AllocationConfig):
    type: Literal[AllocationStrategy.ADMM_CUT] = AllocationStrategy.ADMM_CUT
    cut_type: CutType = defaults.CUT_TYPE
    splits_count: int = defaults.CUT_V_COUNT


class ContagionCutConfig(AllocationConfig):
    type: Literal[AllocationStrategy.ADMM_CONTAGION_CUT] = AllocationStrategy.ADMM_CONTAGION_CUT
    cut_type: CutType = defaults.CUT_TYPE
    splits_count: int = defaults.CUT_V_COUNT
    ridge: float = 1e-8


AllocationConfigUnion = Annotated[
    Union[
        Annotated[CutConfig, Tag(AllocationStrategy.ADMM_CUT.name)],
        Annotated[ContagionCutConfig, Tag(AllocationStrategy.ADMM_CONTAGION_CUT.name)],
    ],
    Discriminator(strategy_discriminator),
]

AllocationConfigAdapter = TypeAdapter(AllocationConfigUnion)
