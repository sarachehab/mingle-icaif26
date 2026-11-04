import numpy as np
import logging

from functools import wraps
from typing import Callable
from inspect import signature
from mingle.utils.runner.loader.experiment import ExperimentLoader, LoadedWindow


WindowAccessorLogger = logging.getLogger(__name__)


def use_per_window_metric(metric_fn: Callable):

    sig = signature(metric_fn)
    param_names = list(sig.parameters)

    def resolve_window_arg(loaded_window: LoadedWindow, name: str):
        if hasattr(loaded_window, name):
            return getattr(loaded_window, name)

        if hasattr(loaded_window.info, name):
            return getattr(loaded_window.info, name)

        if name in loaded_window.history.columns:
            return loaded_window.history[name]

        WindowAccessorLogger.info(f"Could not find attribute {name} in loaded window.")

    @wraps(metric_fn)
    def wrapper(*args, loader: ExperimentLoader, **kwargs):
        metrics = []

        for loaded_window in loader.iter_windows():

            injected = {
                name: resolve_window_arg(loaded_window, name)
                for name in param_names
                if name not in kwargs
            }

            metric = metric_fn(
                *args,
                **injected, # loader is injected here
                **kwargs,
            )

            metrics.append(metric)

        return metrics

    return wrapper
