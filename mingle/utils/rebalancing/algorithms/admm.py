import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from mingle.utils.admm import AdmmJointAlgorithm, AdmmNoLowRankAlgorithm
from mingle.utils.admm import get_representation_cov, get_representation_cov_nolowrank, get_observations_decay
from mingle.utils.rebalancing.algorithms.abstract import ModelWrapper, RebalanceDebug
from mingle.utils.allocation import AbstractAllocation

@dataclass
class AdmmIteration(RebalanceDebug):
    W: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None
    history: Optional[List] = None

class AdmmRebalancingWrapper(ModelWrapper):

    def __init__(self,
        nb_factors,
        alpha, beta, lambda_l, eps, delta,
        max_epochs, abs_eps, rel_eps, abs_eps_B, rel_eps_B,
        rho_B, rho_W, rho_d,
        allocation: AbstractAllocation,
        adapt_every = 25, mu = 10.0, tau = 2.0, rho_min = 1e-3, rho_max = 1e3,
        decay=1.0,
        use_low_rank_prior=True,
        seed: int = 0,
        noise_scale: float = 0.0,
    ):
        super().__init__(noise_scale=noise_scale, seed=seed)
        self.use_low_rank_prior = use_low_rank_prior
        if use_low_rank_prior:
            self.admm = AdmmJointAlgorithm(
                nb_factors=nb_factors,
                alpha=alpha, beta=beta, lambda_l=lambda_l, eps=eps, delta=delta,
                max_epochs=max_epochs, abs_eps=abs_eps, rel_eps=rel_eps, abs_eps_B=abs_eps_B, rel_eps_B=rel_eps_B,
                adapt_every=adapt_every, mu=mu, tau=tau,
                rho_min=rho_min, rho_max=rho_max,
            )
        else:
            self.admm = AdmmNoLowRankAlgorithm(
                alpha=alpha, beta=beta, lambda_l=lambda_l, eps=eps,
                max_epochs=max_epochs, abs_eps=abs_eps, rel_eps=rel_eps,
                adapt_every=adapt_every, mu=mu, tau=tau,
                rho_min=rho_min, rho_max=rho_max,
            )
        self.rho_B, self.rho_W, self.rho_d = rho_B, rho_W, rho_d
        self.decay = decay
        self.allocation = allocation

    def _get_omega(self, X_train):
        return get_observations_decay(self.decay, X_train.shape[1])

    def _get_cov(self, X_train, omega, F=None, B=None, Y=None):
        if self.use_low_rank_prior:
            _, _, cov = get_representation_cov(X_train, B, F, omega)
            return cov
        return get_representation_cov_nolowrank(Y, omega)

    def get_weights(self, X_train, asset_names=None):

        X_train = self.add_noise(X_train)
        omega = self._get_omega(X_train)

        if self.use_low_rank_prior:
            # Initialise values
            F, B, Q, W, d, V, dual_B, dual_W, dual_d = self.admm.initialise(X_train, omega)

            # Run ADMM
            F, B, W, _, history, terminates = self.admm.run(
                X=X_train,
                F=F, B=B, Q=Q, W=W, d=d, V=V, dual_B=dual_B, dual_W=dual_W, dual_d=dual_d,
                omega=omega,
                rho_B=self.rho_B, rho_W=self.rho_W, rho_d=self.rho_d,
            )

            # Build covariance
            cov = self._get_cov(X_train, omega, F=F, B=B)

            # Allocate capital
            weights = self.allocation.get_weights(cov=cov, W=W)
            return (
                weights,
                AdmmIteration(
                    F=F, B=B, W=W,
                    history=history,
                    terminates=terminates,
                ),
            )

        else:
            # Initialise values
            Y, W, d, V, dual_W, dual_d = self.admm.initialise(X_train, omega)

            # Run ADMM
            Y, W, _, history, terminates = self.admm.run(
                X=X_train,
                Y=Y, W=W, d=d, dual_W=dual_W, dual_d=dual_d,
                omega=omega,
                rho_W=self.rho_W, rho_d=self.rho_d,
            )

            # Build covariance
            cov = self._get_cov(X_train, omega, Y=Y)

            # Allocate capital
            weights = self.allocation.get_weights(cov=cov, W=W)
            return (
                weights,
                AdmmIteration(
                    Y=Y, W=W,
                    history=history,
                    terminates=terminates,
                ),
            )

    def backtest(self, X_train: np.ndarray, data: AdmmIteration):

        X_train = self.add_noise(X_train)
        omega = self._get_omega(X_train)

        cov = self._get_cov(X_train, omega, F=data.F, B=data.B, Y=data.Y)
        weights = self.allocation.get_weights(cov=cov, W=data.W)

        return weights
