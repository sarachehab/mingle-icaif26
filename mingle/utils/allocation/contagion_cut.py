import numpy as np

from mingle.utils.allocation.abstract import AbstractAllocation


class ContagionCut(AbstractAllocation):

    def __init__(self, cut, ridge: float = 1e-8):
        self.cut = cut
        self.ridge = ridge

    def get_weights(self, cov, W):
        # Defensive symmetrize W
        W_sym = 0.5 * (W + W.T)
        np.fill_diagonal(W_sym, 0.0)
        W_sym = np.clip(W_sym, 0, None)

        def get_multiplier(idx, W_sym):

            W_leaf = W_sym[np.ix_(idx, idx)]
            local_degree = np.abs(W_leaf).sum(axis=1)

            if local_degree.sum() < 1e-10:
                return np.ones(len(idx))

            return 1.0 / (local_degree + self.ridge)

        weights, _ = self.cut.get_weights(W=cov, get_multiplier=get_multiplier, W_sym=W_sym)
        return weights
