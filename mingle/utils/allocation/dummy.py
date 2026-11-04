import numpy as np
from mingle.utils.allocation.abstract import AbstractAllocation

class DummyAllocation(AbstractAllocation):

    def __init__(self):
        pass

    def get_weights(self, cov, W):
        N, _ = cov.shape
        return np.ones(N) / N
