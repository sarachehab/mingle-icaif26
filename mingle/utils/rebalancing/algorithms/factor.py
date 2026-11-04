import numpy as np
from dataclasses import dataclass, KW_ONLY
from mingle.utils.rebalancing.algorithms.abstract import ModelWrapper, RebalanceDebug
from mingle.utils.admm.helpers import get_observations_decay

@dataclass
class FactorModelIteration(RebalanceDebug):
    _: KW_ONLY
    cut_eigenvals: list

class FactorWrapperModel(ModelWrapper):

    def __init__(
        self,
        nb_factors: int,
        cut,
        decay: float = 1.0,
        seed: int = 0,
        noise_scale: float = 0.0,
    ):
        super().__init__(noise_scale=noise_scale, seed=seed)
        self.nb_factors = nb_factors
        self.decay = decay
        self.cut = cut

    def get_weights(self, X_train, asset_names=None):

        X_train = self.add_noise(X_train)
        _, T = X_train.shape

        omega = get_observations_decay(self.decay, T)

        # Centre observations
        X_centered = X_train - np.average(X_train, axis=1, weights=omega)[:, None]

        # Extract top components of X_train
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        X_top_components = (
            U[:, :self.nb_factors]
            @ (S[:self.nb_factors, None] * Vt[:self.nb_factors, :])
        )

        # Run spectral cut
        cov = np.cov(X_top_components, bias=False, aweights=omega)

        weights, cut_eigenvals = self.cut.get_weights(
            W=cov,
            get_multiplier=lambda idx: np.ones(len(idx)),
        )

        return (
            weights,
            FactorModelIteration(cut_eigenvals=cut_eigenvals),
        )
