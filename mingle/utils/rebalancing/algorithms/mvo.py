import cvxpy as cp
import numpy as np
from dataclasses import dataclass, KW_ONLY
from mingle.utils.rebalancing.algorithms.abstract import ModelWrapper, RebalanceDebug
from mingle.utils.admm.helpers import get_observations_decay

@dataclass
class MinimumVarianceIteration(RebalanceDebug):
    _: KW_ONLY
    C: np.ndarray

class MinimumVarianceWrapper(ModelWrapper):
    def __init__(self,
        decay: float = 1.0,
        ridge=1e-8,
        seed: int = 0,
        noise_scale: float = 0.0,
    ):
        super().__init__(noise_scale=noise_scale, seed=seed)
        self.ridge = ridge
        self.decay = decay

    def get_weights(self, X_train, asset_names=None):

        X_train = self.add_noise(X_train)

        n, T = X_train.shape
        omega = get_observations_decay(self.decay, T)

        C = np.cov(X_train, bias=False, aweights=omega)
        C = C + self.ridge * np.eye(C.shape[0])
        n = C.shape[0]

        w = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(w, C)),
            [cp.sum(w) == 1, w >= 0]
        )
        prob.solve()

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MVO failed: {prob.status}")

        return (np.array(w.value), MinimumVarianceIteration(C=C, terminates=True))
