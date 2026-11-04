from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from textwrap import fill

from mingle.utils.enums import ADMM_STRATEGIES
from mingle.utils.visualisations.configuration import LegendTitle


# ── VIX-regime colours ────────────────────────────────────────────────────────
_REGIME_COLORS = {
    "Calm":     "none",
    "Elevated": "#f3db85",
    "Crisis":   "#ffa398",
}
_REGIME_ORDER = ("Calm", "Elevated", "Crisis")


def smooth_regime_bands(regimes: pd.Series, min_run: int = 15) -> pd.Series:
    """
    Remove short regime runs for plotting only.

    Any contiguous run shorter than `min_run` observations is replaced by the
    previous regime where possible, otherwise by the next regime.
    """
    if regimes is None or regimes.empty:
        return regimes

    s = regimes.dropna().astype(str).copy()

    values = s.to_numpy()
    idx = s.index

    change_idx = np.where(values[1:] != values[:-1])[0] + 1
    boundaries = np.concatenate([[0], change_idx, [len(values)]])

    smoothed = values.copy()

    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        run_len = hi - lo

        if run_len >= min_run:
            continue

        if lo > 0:
            replacement = smoothed[lo - 1]
        elif hi < len(values):
            replacement = values[hi]
        else:
            replacement = values[lo]

        smoothed[lo:hi] = replacement

    return pd.Series(smoothed, index=idx, name=regimes.name)


def _shade_regimes(ax: Axes, regimes: pd.Series) -> None:
    """
    Fill background of ax with VIX-regime colour bands.

    For clean solid bands on monthly-sampled plots, pre-process
    the input with `smooth_regime_bands`.
    """
    regimes = smooth_regime_bands(regimes)
    if regimes is None or regimes.empty:
        return

    regimes = regimes.dropna()
    if regimes.empty:
        return

    regime_arr = regimes.astype(str).to_numpy()
    dates_arr = pd.to_datetime(regimes.index).to_numpy()

    change_idx = np.where(regime_arr[1:] != regime_arr[:-1])[0] + 1
    boundaries = np.concatenate([[0], change_idx, [len(regime_arr)]])

    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        label = regime_arr[lo]
        color = _REGIME_COLORS.get(label, "white")

        # Right boundary: start of next run (gap-free), or last date for the
        # final run.
        right = dates_arr[hi] if hi < len(dates_arr) else dates_arr[-1]

        ax.axvspan(
            dates_arr[lo],
            right,
            alpha=0.35,
            color=color,
            zorder=0,
            linewidth=0,
        )


def _regime_patches(regimes: pd.Series) -> list[Patch]:
    """Legend patches matching the _shade_regimes background bands."""
    present = set(regimes.dropna().values)
    patches = []
    for r in _REGIME_ORDER:
        if r not in present:
            continue
        if r == "Calm":
            patches.append(Patch(facecolor="white", edgecolor="black", linewidth=1, label="Calm"))
        else:
            patches.append(Patch(facecolor=_REGIME_COLORS[r], alpha=0.35, label=r))
    return patches


def _sectioned_legend(ax: Axes, sections: list[tuple[str, list]], **legend_kwargs):
    """
    Legend with one column per section, each headed by a bold subtitle
    (LegendTitle). `sections` is [(title, handles)], handles carrying labels.
    Shorter columns are padded with blank rows so every section stays in its
    own column.
    """
    sections = [(title, handles) for title, handles in sections if handles]
    n_rows = 1 + max(len(handles) for _, handles in sections)
    all_handles, all_labels = [], []
    for title, handles in sections:
        padding = n_rows - 1 - len(handles)
        all_handles += [title] + handles + [""] * padding
        all_labels  += [""] + [h.get_label() for h in handles] + [""] * padding
    return ax.legend(
        all_handles, all_labels, ncol=len(sections),
        handler_map={str: LegendTitle()}, **legend_kwargs,
    )


def add_regime_legend(ax: Axes, regimes: pd.Series) -> None:
    """
    Attach a 'Market volatility' section-header legend to ax.

    Call this after all lines have been plotted so existing handles are
    captured correctly.
    """
    patches = _regime_patches(regimes)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + ["Market volatility"] + patches,
        labels  + [""]                  + [p.get_label() for p in patches],
        handler_map={str: LegendTitle()},
    )


def plot_regime_shaded_returns(
    results: dict,
    regimes: pd.Series,
    start_test=None,
    ax=None,
    plot_specs: list | None = None,
    legend: bool = True,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Cumulative returns with VIX-regime background shading.

    Args:
        results : dict
        regimes : pd.Series
        start_test : optional date
        ax : optional matplotlib Axes
        plot_specs : optional list of (model, color, linestyle, linewidth)
        legend : bool
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    lines = {}
    if plot_specs is not None:
        for alg, color, linestyle, linewidth in plot_specs:
            ret = results[alg].returns
            if start_test is not None:
                ret = ret.loc[start_test:]
            lines[alg], = ax.plot((1 + ret).cumprod(), label=alg.value,
                                  color=color, linestyle=linestyle, linewidth=linewidth)
    else:
        for alg, content in results.items():
            ret = content.returns
            if start_test is not None:
                ret = ret.loc[start_test:]
            lines[alg], = ax.plot((1 + ret).cumprod(), label=alg.value, linewidth=0.8)

    regime_slice = regimes
    if start_test is not None:
        regime_slice = regimes.loc[start_test:]

    _shade_regimes(ax, regime_slice)

    if legend:
        _sectioned_legend(
            ax,
            [
                ("Learnt representation", [h for alg, h in lines.items() if alg in ADMM_STRATEGIES]),
                ("Benchmarks", [h for alg, h in lines.items() if alg not in ADMM_STRATEGIES]),
                ("Market volatility", _regime_patches(regime_slice)),
            ],
            loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False,
        )

    ax.set_ylabel(r"Cumulative returns")
    ax.set_xlabel("Date")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_performance_by_regime_bars(
    results: dict,
    regimes: pd.Series,
    rf_returns: pd.Series,
    market_returns: pd.Series,
    ax=None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart: Sharpe / Calmar per strategy, faceted by VIX regime.

    Args:
        results : dict
        regimes : pd.Series
        rf_returns : pd.Series
        market_returns : pd.Series
        ax : optional array-like of two matplotlib Axes (one per metric)
    """
    from mingle.utils.metrics.regime import performance_by_regime as _perf_by_regime

    algs = list(results.keys())
    regime_labels = [r for r in _REGIME_ORDER if r in regimes.values]
    metrics_keys   = ["annualised_sharpe_ratio"]
    metrics_titles = ["Sharpe Ratio"]

    data = {}
    for alg in algs:
        ret = results[alg].returns
        data[alg] = _perf_by_regime(ret, rf_returns, market_returns, regimes)

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(8, 19))
    else:
        axes = ax
        fig = axes[0].get_figure()

    n_algs    = len(algs)
    n_regimes = len(regime_labels)
    width = 0.8 / n_regimes
    x = np.arange(n_algs) * 2.0
    wrapped_labels = [fill(str(a), width=15) for a in algs]

    for col, (metric, title) in enumerate(zip(metrics_keys, metrics_titles)):
        for r_idx, regime in enumerate(regime_labels):
            values = []
            for alg in algs:
                pm = data[alg].get(regime)
                values.append(getattr(pm, metric) if pm is not None else float("nan"))

            offset = (r_idx - n_regimes / 2 + 0.5) * width

            bars = axes[col].bar(
                x + offset, values,
                width=width,
                label=regime,
                color=_REGIME_COLORS[regime],
                edgecolor="black",
                linewidth=0.5,
            )

            for bar in bars:
                height = bar.get_height()
                if np.isnan(height):
                    continue
                if height >= 0:
                    va, xytext = "bottom", (0, 3)
                else:
                    va, xytext = "top", (0, -3)

                axes[col].annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=xytext,
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6),
                    clip_on=False,
                )

        axes[col].relim()
        axes[col].autoscale_view()

        ymin, ymax = axes[col].get_ylim()
        yrange = ymax - ymin
        pad = 0.18 * yrange
        axes[col].set_ylim(ymin - pad, ymax + pad)

        if col != len(metrics_keys) - 1:
            axes[col].tick_params(axis="x", labelbottom=False)
        else:
            axes[col].set_xticks(x)
            axes[col].set_xticklabels(wrapped_labels, rotation=20, ha="center")
        axes[col].axhline(0, color="black", linewidth=0.8)
        axes[col].set_ylabel(title)
        if col == 0:
            handles, labels = axes[col].get_legend_handles_labels()
            axes[col].legend(
                ["Market volatility"] + handles, [""] + labels,
                handler_map={str: LegendTitle()},
                loc="lower center", bbox_to_anchor=(0.5, 1.02),
                ncol=1 + n_regimes, frameon=False,
            )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
