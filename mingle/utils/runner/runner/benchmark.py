from pathlib import Path

from mingle.utils.enums import AllocationStrategy
from mingle.utils.runner.input import Config
from mingle.utils.rebalancing import (
    EquallyWeightedWrapper,
    MinimumVarianceWrapper,
    HrpRebalancingWrapper,
    CutWrapperModel,
    FactorWrapperModel,
)
from mingle.utils.rebalancing.algorithms import ModelWrapper
from mingle.utils.runner.runner.base import BaseRunner
from mingle.utils.allocation.cut import get_cut_alg
from mingle.utils.runner.output.constants import BENCHMARKS_DIR


class BenchmarkRunner(BaseRunner):

    def __init__(
        self,
        config: Config,
        output_root: Path,
        git_hash: str,
    ):
        super().__init__(
            config=config,
            output_root=output_root / BENCHMARKS_DIR,
            git_hash=git_hash,
            save_arrays=False,
            save_window=False,
            save_perf=True
        )

    def build_model(self) -> ModelWrapper:

        benchmark_config = self.config.benchmark
        wrapper_type = benchmark_config.type
        kwargs = benchmark_config.model_dump(exclude={"type", "ablation"})

        seed = self.config.experiment.seed
        noise_scale = self.config.experiment.noise_scale

        match wrapper_type:

            case AllocationStrategy.MVO: return MinimumVarianceWrapper(seed=seed, noise_scale=noise_scale, **kwargs)

            case AllocationStrategy.EW: return EquallyWeightedWrapper(**kwargs)

            case AllocationStrategy.HRP: return HrpRebalancingWrapper(seed=seed, noise_scale=noise_scale, **kwargs)

            case AllocationStrategy.CUT:

                cut_alg = get_cut_alg(
                    cut_type=benchmark_config.cut_type,
                    splits_count=benchmark_config.splits_count,
                )

                kwargs = benchmark_config.model_dump(
                    exclude={"type", "ablation", "cut_type", "splits_count"}
                )

                return CutWrapperModel(cut=cut_alg, seed=seed, noise_scale=noise_scale, **kwargs)
            
            case AllocationStrategy.FACTOR:

                cut_alg = get_cut_alg(
                    cut_type=benchmark_config.cut_type,
                    splits_count=benchmark_config.splits_count,
                )

                kwargs = benchmark_config.model_dump(
                    exclude={"type", "ablation", "cut_type", "splits_count"}
                )

                return FactorWrapperModel(nb_factors=benchmark_config.nb_factors, cut=cut_alg, seed=seed, noise_scale=noise_scale)


        raise NotImplementedError(
            f"Benchmark for model wrapper {wrapper_type} not implemented in BenchmarkRunner."
        )
