import numpy as np
from mingle.utils.allocation.abstract import AbstractAllocation

class CutAllocation(AbstractAllocation):

    def __init__(self, cut):
        self.cut = cut

    def get_weights(self, cov, W):

        weights, _ = self.cut.get_weights(
            W = cov,
            get_multiplier = lambda idx: np.ones(len(idx)),
        )

        return weights
