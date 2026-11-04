from __future__ import annotations

from pydantic import BaseModel, TypeAdapter, field_validator
from typing import Any, Optional, Generator
from mingle.utils.enums import AllocationStrategy, coerce_strategy
from mingle.utils.runner.input.ablation import AblationConfig


class BaseConfig(BaseModel):

    ablation: Optional[AblationConfig] = None

    def expand_runs(self) -> Generator[BaseConfig, None, None]:

        if self.ablation is None:
            yield self
            return

        for i, override in enumerate(self.ablation.overrides):
            yield self.apply_overrides(override)

    def apply_overrides(self, override: dict[str, Any]) -> BaseConfig:

        data = self.model_dump()

        for dotted_key, value in override.items():
            parts = dotted_key.split(".")
            target = data

            for part in parts[:-1]:
                target = target[part]

            target[parts[-1]] = value

        return self.__class__.model_validate(data)


def strategy_discriminator(v) -> str:
    raw = v.get("type") if isinstance(v, dict) else getattr(v, "type", None)
    try: return coerce_strategy(raw).name
    except (ValueError, TypeError): return raw


class StrategyConfig(BaseConfig):

    type: AllocationStrategy

    @field_validator("type", mode="before")
    @classmethod
    def _accept_short_code(cls, v):
        return coerce_strategy(v)

    @classmethod
    def _adapter(cls) -> TypeAdapter:
        raise NotImplementedError

    def apply_overrides(self, override: dict[str, Any]) -> "StrategyConfig":
        data = self.model_dump(exclude={"ablation"})
        for key, value in override.items():
            data[key] = value
        return self._adapter().validate_python(data)
