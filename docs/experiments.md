# Reproducing the paper

The universe is 300 equities: the 100 most actively traded constituents of each of the **S&P 500**, **Nikkei 225** and **STOXX 600**, with daily USD close prices from 2017-01-04 to 2026-03-30. The representation is re-estimated monthly on a two-year lookback and evaluated on the following month, giving a held-out period of January 2019 to March 2026.

> [!IMPORTANT]
> **The equity prices were extracted from the Bloomberg Terminal and cannot be redistributed.** With Bloomberg Terminal access, [data/bloomberg_extractor.ipynb](../data/bloomberg_extractor.ipynb) extracts them in the format expected by the loader. Otherwise, any other data source may be used, as described in [Using your own data](#using-your-own-data). The risk-free rate and VIX series are publicly available.

## Using your own data

Data from another source can be loaded by implementing `CustomDataFetcher` in [mingle/utils/fetcher/custom.py](../mingle/utils/fetcher/custom.py), and setting `custom_path` in the `data` section of a config:

```yaml
data:
  custom_path: path/to/your/data   # passed to every CustomDataFetcher method as `filepath`
  start_raw: 2017-01-04
  end_raw: 2026-03-30
```

## Running the experiment suite

Each experiment is a YAML config in [experiments/](../experiments/) that lists only its overrides of [mingle/configs/defaults.py](../mingle/configs/defaults.py). Configs have six sections (`experiment`, `data`, `rebalancing`, `admm`, `benchmark`, `allocation`), and any section can carry an `ablation` list, which the runner expands into one run per override. For example, [experiments/ref.yaml](../experiments/ref.yaml) compares both MINGLE allocations against every baseline:

```yaml
experiment:
  name: reference
  description: "Standard run"

benchmark:
  type: EW
  ablation:
    name: benchmarks
    overrides:
      - type: EW
      - type: MVO
      - type: HRP
      - type: CUT
      - type: FACTOR

allocation:
  type: ADMM_CUT
  ablation:
    name: all_allocations
    overrides:
      - type: ADMM_CUT
      - type: ADMM_CONTAGION_CUT
```

To depart from the default hyperparameters, the corresponding field is set in its section. For instance, the following increases the sparsity penalty, $\alpha$, and the number of factors, $k$:

```yaml
admm:
  alpha: 3.0
  nb_factors: 8
```

The fields available in each section are those of its config class in [mingle/utils/runner/input/](../mingle/utils/runner/input/), e.g. `AdmmConfig` for `admm`.

A hyperparameter sweep is expressed as an `ablation`, with one override per value. Fields set outside the ablation are shared by every run, and a single override may change several fields at once:

```yaml
admm:
  nb_factors: 8              # shared by every run
  ablation:
    name: alpha_sweep
    overrides:
      - alpha: 1.0
      - alpha: 2.2
      - {alpha: 3.0, beta: 2.0}
```

> [!NOTE]
> Ablations in different sections combine as a grid. For instance, every `admm` run is evaluated under every `allocation` override.

Both suites of the paper are run with:

```bash
run-experiment -c experiments/ref.yaml     # 300 assets: Table 2, Figures 2, 4, 5 and 6
run-experiment -c experiments/sp500.yaml   # S&P 500 only: Figure 3
```

> [!TIP]
> Passing a directory, `-c experiments/`, runs every config it contains. Outputs are written to `runs/experiments/<config>/`. For every rebalancing window, these include the learnt $\mathbf B$, $\mathbf F$ and $\mathbf W$, together with the returns and weights of each strategy.

## Tables and figures

Once both suites have been run, refer to [visualisations.ipynb](../visualisations.ipynb) to reproduce the tables and figures of the paper. Each is saved under `runs/experiments/<config>/figures/`:

| Paper | Output | Config |
|---|---|---|
| **Table 1**: hyperparameters | [mingle/configs/defaults.py](../mingle/configs/defaults.py) | – |
| **Table 2**: performance under transaction costs | `table2.csv` | `ref.yaml` |
| **Figure 2**: temporal stability of the learnt graph (Jun–Nov 2022) | `adjacency_from_history.pdf` | `ref.yaml` |
| **Figure 3**: learnt vs. correlation graph, ordered by GICS sector | `adjacency_sector_boundaries.pdf` | `sp500.yaml` |
| **Figure 4**: conditioning of $\hat{\mathbf\Sigma}_R$ vs. $\hat{\mathbf\Sigma}_S$ | `cov_condition_evolution.pdf` | `ref.yaml` |
| **Figure 5**: cumulative returns across VIX regimes | `regime_shaded_returns.pdf` | `ref.yaml` |
| **Figure 6**: Sharpe ratio within each VIX regime | `performance_by_regime_bars.pdf` | `ref.yaml` |

---
