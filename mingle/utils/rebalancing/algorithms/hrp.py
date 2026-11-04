import pandas as pd
import numpy as np
from pypfopt.hierarchical_portfolio import HRPOpt
from dataclasses import dataclass
from mingle.utils.rebalancing.algorithms.abstract import ModelWrapper, RebalanceDebug
from mingle.utils.admm.helpers import get_observations_decay

@dataclass
class HrpIteration(RebalanceDebug):
    pass

class HrpRebalancingWrapper(ModelWrapper):
    
    def __init__(
        self,
        decay: float = 1.0,
        linkage_method: str = "single",
        ridge: float = 1e-8,
        seed: int = 0,
        noise_scale: float = 0.0,
    ):
        super().__init__(noise_scale=noise_scale, seed=seed)
        self.decay = decay
        self.linkage_method = linkage_method
        self.ridge = ridge

    def get_weights(self, X_train: np.ndarray, asset_names=None) -> np.ndarray:

        X_train = self.add_noise(X_train)

        n, T = X_train.shape
        omega = get_observations_decay(self.decay, T)

        if asset_names is None:
            asset_names = list(range(n))

        C = np.cov(X_train, bias=False, aweights=omega)
        C = C + self.ridge * np.eye(n)

        cov_matrix = pd.DataFrame(
            C,
            index=asset_names,
            columns=asset_names,
        )

        hrp = HRPOpt(cov_matrix=cov_matrix)
        weights_dict = hrp.optimize(linkage_method=self.linkage_method)

        return (
            np.array([weights_dict[name] for name in asset_names], dtype=float),
            HrpIteration(),
        )
