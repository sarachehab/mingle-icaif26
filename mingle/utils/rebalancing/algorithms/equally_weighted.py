import numpy as np
from dataclasses import dataclass
from mingle.utils.rebalancing.algorithms.abstract import ModelWrapper, RebalanceDebug

@dataclass
class EquallyWeightedIteration(RebalanceDebug):
    pass

class EquallyWeightedWrapper(ModelWrapper):

    def __init__(self):
        pass

    def get_weights(self, X_train, asset_names=None):
        N = X_train.shape[0]
        return (
            np.ones(N,) / N,
            EquallyWeightedIteration(),
        )
