import json
import numpy as np

from pathlib import Path

from mingle.utils.enums import AllocationStrategy
from mingle.utils.runner.input import Config
from mingle.utils.allocation import CutAllocation, ContagionCut
from mingle.utils.rebalancing import AdmmRebalancingWrapper
from mingle.utils.rebalancing.algorithms import ModelWrapper
from mingle.utils.rebalancing.algorithms.admm import AdmmIteration
from mingle.utils.runner.runner.base import BaseRunner
from mingle.utils.allocation.cut import get_cut_alg
from mingle.utils.runner.output.constants import (
    ALLOCATION_DIR, WINDOWS_DIR, WINDOWS_RE,
    WINDOW_FILE, F_FILE, B_FILE, Y_FILE, W_FILE,
)
from mingle.utils.runner.output.window import WindowInfo


class AllocationRunner(BaseRunner):

    def __init__(
        self,
        config: Config,
        output_root: Path,
        git_hash: str,
    ):
        # Allocation outputs nested inside a representation directory
        super().__init__(
            config=config,
            output_root=output_root / ALLOCATION_DIR,
            git_hash=git_hash,
            save_arrays=False,
            save_window=False,
            save_perf=True
        )

    def build_model(self) -> ModelWrapper:
        allocation = self._build_allocation()
        admm_config = self.config.admm

        return AdmmRebalancingWrapper(
            nb_factors=admm_config.nb_factors,
            alpha=admm_config.alpha,
            beta=admm_config.beta,
            lambda_l=admm_config.lambda_l,
            eps=admm_config.eps,
            delta=admm_config.delta,
            max_epochs=admm_config.max_epochs,
            abs_eps=admm_config.abs_eps,
            rel_eps=admm_config.rel_eps,
            abs_eps_B=admm_config.abs_eps_B,
            rel_eps_B=admm_config.rel_eps_B,
            rho_B=admm_config.rho_B,
            rho_W=admm_config.rho_W,
            rho_d=admm_config.rho_d,
            adapt_every=admm_config.adapt_every,
            mu=admm_config.mu,
            tau=admm_config.tau,
            rho_min=admm_config.rho_min,
            rho_max=admm_config.rho_max,
            decay=admm_config.decay,
            use_low_rank_prior=admm_config.use_low_rank_prior,
            allocation=allocation,
            seed=self.config.experiment.seed,
            noise_scale=self.config.experiment.noise_scale,
        )

    def _build_allocation(self):
        allocation_config = self.config.allocation
        wrapper_type = allocation_config.type
        kwargs = allocation_config.model_dump(exclude={"type", "ablation"})

        cut_alg = self._build_cut_alg()
        kwargs = allocation_config.model_dump(
            exclude={"type", "ablation", "cut_type", "splits_count"}
        )

        match wrapper_type:

            case AllocationStrategy.ADMM_CUT:
                return CutAllocation(cut=cut_alg, **kwargs)
            
            case AllocationStrategy.ADMM_CONTAGION_CUT:
                return ContagionCut(cut=cut_alg, **kwargs)

        raise NotImplementedError(
            f"Allocation for model wrapper {wrapper_type} not implemented in AllocationRunner."
        )

    def run(self) -> None:
        np.random.seed(self.config.experiment.seed)

        saver = self.build_saver()
        saver.save_config(config=self.config)

        prices_df = self.load_data()
        dataset_metadata = self.build_dataset_metadata(prices_df)
        saver.save_metadata(dataset=dataset_metadata)

        model = self.build_model()
        rebalancing = self.build_rebalancing()

        in_ret, out_ret, portfolio_weights = rebalancing.backtest(
            prices_df=prices_df,
            model_wrapper=model,
            per_rebalance_data=self._get_rebalance_data(),
        )

        saver.save_performance(in_ret=in_ret, out_ret=out_ret, weights=portfolio_weights)

    def _get_rebalance_data(self) -> list[AdmmIteration]:
        windows_root = self.output_root.parent / WINDOWS_DIR
        use_low_rank = self.config.admm.use_low_rank_prior

        window_dirs = sorted(
            [d for d in windows_root.iterdir() if WINDOWS_RE.match(d.name)],
            key=lambda d: int(WINDOWS_RE.match(d.name).group(1)),
        )

        iterations = []
        for window_dir in window_dirs:
            with (window_dir / WINDOW_FILE).open() as f:
                window_info = WindowInfo.model_validate(json.load(f))

            W = np.load(window_dir / W_FILE)

            if use_low_rank:
                F = np.load(window_dir / F_FILE)
                B = np.load(window_dir / B_FILE)
                iteration = AdmmIteration(
                    F=F, B=B, W=W,
                    history=[],
                    terminates=window_info.terminates,
                    start_train=window_info.train_start,
                    end_train=window_info.train_end,
                    start_test=window_info.test_start,
                    end_test=window_info.test_end,
                    duration_seconds=window_info.duration_seconds,
                )
            else:
                Y = np.load(window_dir / Y_FILE)
                iteration = AdmmIteration(
                    Y=Y, W=W,
                    history=[],
                    nb_clusters=window_info.nb_clusters,
                    terminates=window_info.terminates,
                    start_train=window_info.train_start,
                    end_train=window_info.train_end,
                    start_test=window_info.test_start,
                    end_test=window_info.test_end,
                    duration_seconds=window_info.duration_seconds,
                )

            iterations.append(iteration)

        return iterations

    def _build_cut_alg(self):
        allocation_config = self.config.allocation
        return get_cut_alg(
            cut_type=allocation_config.cut_type,
            splits_count=allocation_config.splits_count,
        )
