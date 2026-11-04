from pathlib import Path

import numpy as np
from mingle.utils.rebalancing import AdmmRebalancingWrapper
from mingle.utils.allocation import DummyAllocation
from mingle.utils.runner.runner.base import BaseRunner
from mingle.utils.runner.input.config import Config
from mingle.utils.runner.output.constants import REPRESENTATION_DIR

class RepresentationLearningRunner(BaseRunner):

    def __init__(
        self,
        config: Config,
        output_root: Path,
        git_hash: str,
        save_arrays: bool,
    ):
        super().__init__(
            config=config,
            output_root=output_root / REPRESENTATION_DIR,
            git_hash=git_hash,
            save_arrays=save_arrays,
            save_window=True,
            save_perf=False,
        )

    def build_model(self) -> AdmmRebalancingWrapper:
        model_config = self.config.admm
        seed = self.config.experiment.seed
        return AdmmRebalancingWrapper(
            nb_factors=model_config.nb_factors,
            alpha=model_config.alpha, beta=model_config.beta, lambda_l=model_config.lambda_l, eps=model_config.eps, delta=model_config.delta,
            max_epochs=model_config.max_epochs,
            abs_eps=model_config.abs_eps, rel_eps=model_config.rel_eps, abs_eps_B=model_config.abs_eps_B, rel_eps_B=model_config.rel_eps_B,
            rho_B=model_config.rho_B, rho_W=model_config.rho_W, rho_d=model_config.rho_d,
            adapt_every=model_config.adapt_every, mu=model_config.mu, tau=model_config.tau, rho_min=model_config.rho_min, rho_max=model_config.rho_max,
            decay=model_config.decay,
            use_low_rank_prior=model_config.use_low_rank_prior,
            allocation=DummyAllocation(),
            seed=seed,
            noise_scale=self.config.experiment.noise_scale,
        )
