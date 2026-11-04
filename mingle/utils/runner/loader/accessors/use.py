from functools import wraps
from inspect import signature
from typing import Callable
from mingle.utils.runner.loader.experiment import ExperimentLoader


def use_object(get_obj: Callable):
    def decorator(metric_fn: Callable):
        param_names = signature(metric_fn).parameters

        @wraps(metric_fn)
        def wrapper(*args, loader: ExperimentLoader, **kwargs):
            obj = get_obj(loader)

            injected = {
                name: getattr(obj, name)
                for name in param_names
                if name not in kwargs and hasattr(obj, name)
            }

            return metric_fn(
                *args,
                **injected,
                loader=loader,
                **kwargs,
            )

        return wrapper

    return decorator
