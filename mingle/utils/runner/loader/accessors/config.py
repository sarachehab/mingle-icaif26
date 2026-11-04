from mingle.utils.runner.loader.accessors.use import use_object


def use_config_section(section: str):
    return use_object(
        lambda loader: getattr(loader.load_config(), section)
    )

use_experiment_config = use_config_section("experiment")
use_data_config = use_config_section("data")
use_ablation_config = use_config_section("ablation")
use_rebalancing_config = use_config_section("rebalancing")
use_admm_config = use_config_section("admm")
use_allocation_config = use_config_section("allocation")
use_benchmark_config = use_config_section("benchmark")
