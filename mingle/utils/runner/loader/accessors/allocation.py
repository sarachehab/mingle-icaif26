import logging

from functools import wraps
from typing import Callable
from inspect import signature
from mingle.utils.runner.loader.experiment import ExperimentLoader, LoadedAllocation


AllocationAccessorLogger = logging.getLogger(__name__)


def use_allocation_metric(metric_fn: Callable):

    sig = signature(metric_fn)
    param_names = list(sig.parameters)

    def resolve_allocation_arg(loaded_allocation: LoadedAllocation, name: str):
        if hasattr(loaded_allocation, name):
            return getattr(loaded_allocation, name)

        AllocationAccessorLogger.info(f"Could not find attribute {name} in loaded allocation.")

    @wraps(metric_fn)
    def wrapper(*args, loader: ExperimentLoader, **kwargs):
        loaded_allocation = loader.load_allocation()

        injected = {
            name: resolve_allocation_arg(loaded_allocation, name)
            for name in param_names
            if name not in kwargs
        }

        return metric_fn(*args, **injected, **kwargs)

    return wrapper
