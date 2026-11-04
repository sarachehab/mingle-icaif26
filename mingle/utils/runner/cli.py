import argparse
import subprocess
import logging

from pathlib import Path
from typing import Optional

from mingle.utils.runner.input import Config
from mingle.utils.runner.input.data import DataConfig
from mingle.utils.runner.input.experiments import ExperimentConfig
from mingle.utils.runner.input.admm import AdmmConfig
from mingle.utils.runner.input.rebalancing import RebalancingConfig
from mingle.utils.runner.input.benchmark import BenchmarkConfig
from mingle.utils.runner.input.allocation import AllocationConfig
from mingle.utils.runner.runner import RepresentationLearningRunner
from mingle.utils.runner.runner.benchmark import BenchmarkRunner
from mingle.utils.runner.runner.allocation import AllocationRunner
from mingle.configs.constants import RUNS_DIR
from mingle.utils.runner.output.constants import REPRESENTATION_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ADMM representation learning experiments."
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file or directory containing YAML config files.",
    )

    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=RUNS_DIR/"experiments",
        help="Directory where experiment outputs are written.",
    )

    parser.add_argument(
        "--no-arrays",
        action="store_true",
        help="Do not save large NumPy arrays.",
    )

    return parser.parse_args()


def resolve_config_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(path.glob("*.yaml"))

    raise FileNotFoundError(path)


def get_git_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
        

def run_config(config_path: Path, output_root: Path, save_arrays: bool, git_hash: str) -> Path:
    """Run every expanded configuration of one YAML file. Returns its experiment root."""

    cfg = Config.from_yaml(config_path)
    experiment_root = output_root / config_path.stem

    experiment_cfg: ExperimentConfig = cfg.experiment
    data_cfg: DataConfig = cfg.data
    rebalancing_cfg: RebalancingConfig = cfg.rebalancing
    admm_cfg: AdmmConfig = cfg.admm
    benchmark_cfg: Optional[BenchmarkConfig] = cfg.benchmark
    allocation_cfg: Optional[AllocationConfig] = cfg.allocation

    for s, experiment_run in enumerate(experiment_cfg.expand_runs()): 
        
        experiment_run_root = experiment_root / f"experiment_run_{s:02d}"

        experiment_name = experiment_run.name

        for i, data_run in enumerate(data_cfg.expand_runs()):

            data_run_root = experiment_run_root / f"data_configuration_{i:02d}"

            for j, rebalancing_run in enumerate(rebalancing_cfg.expand_runs()):

                rebalancing_run_root = data_run_root / f"rebalancing_configuration_{j:02d}"

                if benchmark_cfg is not None:
                    for k, benchmark_run in enumerate(benchmark_cfg.expand_runs()):

                        benchmark_experiment_cfg = experiment_run.model_copy(
                            update={"name": f"{experiment_name}_{k:02d}_{benchmark_run.type.name}"}
                        )

                        benchmark_experiment_run = Config(
                            experiment=benchmark_experiment_cfg,
                            data=data_run,
                            rebalancing=rebalancing_run,
                            admm=admm_cfg,
                            benchmark=benchmark_run,
                        )

                        benchmark_runner = BenchmarkRunner(
                            config=benchmark_experiment_run,
                            output_root=rebalancing_run_root,
                            git_hash=git_hash,
                        )

                        try: benchmark_runner.run()
                        except Exception: logging.exception("Run %s failed — skipping.", benchmark_experiment_run.experiment.name)

                for k, admm_run in enumerate(admm_cfg.expand_runs()):

                    run_experiment_cfg = experiment_run.model_copy(
                        update={"name": f"{experiment_name}_{k:02d}"}
                    )

                    cfg_run = Config(
                        experiment=run_experiment_cfg,
                        data=data_run,
                        rebalancing=rebalancing_run,
                        admm=admm_run,
                    )

                    representation_runner = RepresentationLearningRunner(
                        config=cfg_run,
                        output_root=rebalancing_run_root,
                        save_arrays=save_arrays,
                        git_hash=git_hash,
                    )
                    try: representation_runner.run()
                    except Exception: logging.exception("Run %s failed — skipping.", cfg_run.experiment.name)

                    if allocation_cfg is not None:

                        representation_run_dir = rebalancing_run_root / REPRESENTATION_DIR / run_experiment_cfg.name

                        for l, allocation_run in enumerate(allocation_cfg.expand_runs()):

                            allocation_experiment_cfg = experiment_run.model_copy(
                                update={"name": f"{experiment_name}_admm_{l:02d}_{allocation_run.type.name}"}
                            )

                            allocation_experiment_run = Config(
                                experiment=allocation_experiment_cfg,
                                data=data_run,
                                rebalancing=rebalancing_run,
                                admm=admm_run,
                                allocation=allocation_run,
                            )

                            allocation_runner = AllocationRunner(
                                config=allocation_experiment_run,
                                output_root=representation_run_dir,
                                git_hash=git_hash,
                            )

                            try: allocation_runner.run()
                            except Exception: logging.exception("Run %s failed — skipping.", allocation_experiment_run.experiment.name)

    return experiment_root


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config_paths = resolve_config_paths(args.config)
    git_hash = get_git_hash()

    for config_path in config_paths:
        run_config(config_path, args.output_root, save_arrays=not args.no_arrays, git_hash=git_hash)


if __name__ == "__main__":
    main()
