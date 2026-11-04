from pathlib import Path

from mingle.utils.enums import AllocationStrategy
from mingle.utils.runner.output.constants import CONFIG_FILE, BENCHMARKS_DIR, REPRESENTATION_DIR, ALLOCATION_DIR
from mingle.utils.runner.loader.experiment import ExperimentLoader


class ExperimentSuiteLoader:
    """
    Loads across a whole suite directory (a rebalancing_configuration_NN/
    directory containing benchmarks/ and representation/*/allocations/
    subdirs).
    """

    def __init__(self, suite_dir: Path):
        self.suite_dir = Path(suite_dir)

    def strategies(
        self,
        strategies: list[AllocationStrategy] | None = None,
    ) -> dict[AllocationStrategy, ExperimentLoader]:
        """Create a dictionary of all benchmarks and learnings to run."""
        found: dict[AllocationStrategy, ExperimentLoader] = {}

        for cfg_path in sorted(self.suite_dir.glob(f"{BENCHMARKS_DIR}/*/{CONFIG_FILE}")):
            loader = ExperimentLoader(cfg_path.parent)
            found[loader.load_config().benchmark.type] = loader

        for cfg_path in sorted(self.suite_dir.glob(f"{REPRESENTATION_DIR}/*/{ALLOCATION_DIR}/*/{CONFIG_FILE}")):
            loader = ExperimentLoader(cfg_path.parent)
            found[loader.load_config().allocation.type] = loader

        if strategies is not None:
            found = {s: l for s, l in found.items() if s in strategies}

        return found

    def representation_loader(self, name: str | None = None) -> ExperimentLoader:
        candidates = sorted((self.suite_dir / REPRESENTATION_DIR).glob(f"*/{CONFIG_FILE}"))
        if name is not None:
            candidates = [p for p in candidates if p.parent.name == name]

        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one representation run in {self.suite_dir}, "
                f"found {len(candidates)}. Pass `name=` to disambiguate."
            )
        return ExperimentLoader(candidates[0].parent)

    def allocation_loader(self, strategy: AllocationStrategy) -> ExperimentLoader:
        """The run directory (benchmark or ADMM allocation) for one strategy."""
        strategies = self.strategies([strategy])
        if strategy not in strategies:
            raise KeyError(f"Strategy {strategy} not found in suite {self.suite_dir}")
        return strategies[strategy]
