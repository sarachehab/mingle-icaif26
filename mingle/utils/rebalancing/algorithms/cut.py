import numpy as np
from dataclasses import dataclass, KW_ONLY
from mingle.utils.rebalancing.algorithms.abstract import ModelWrapper, RebalanceDebug
from mingle.utils.admm.helpers import get_observations_decay

@dataclass
class CutWrapperModelIteration(RebalanceDebug):
    _: KW_ONLY
    cut_eigenvals: list

class CutWrapperModel(ModelWrapper):

    def __init__(
        self, 
        cut,
        decay: float = 1.0,
        seed: int = 0,
        noise_scale: float = 0.0,
    ):
        super().__init__(noise_scale=noise_scale, seed=seed)
        self.decay = decay
        self.cut = cut

    def get_weights(self, X_train, asset_names=None):
        
        X_train = self.add_noise(X_train)
        _, T = X_train.shape
        omega = get_observations_decay(self.decay, T)

        cov = np.cov(X_train, bias=False, aweights=omega)
        weights, cut_eigenvals = self.cut.get_weights(
            W = cov,
            get_multiplier = lambda idx: np.ones(len(idx))
        )
        
        return (
            weights,
            CutWrapperModelIteration(cut_eigenvals=cut_eigenvals),
        )
