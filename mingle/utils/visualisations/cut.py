from __future__ import annotations

from pathlib import Path
from typing import Optional

from mingle.utils.admm import get_observations_decay, get_representation_cov
from mingle.utils.transforms import log_returns
from mingle.utils.metrics.stability import build_abs_correlation_graph, density_match_correlation
from mingle.utils.runner.loader.experiment import LoadedWindow

import numpy as np
import pandas as pd
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from mingle.utils.allocation.cut.node import Node
from matplotlib.ticker import FuncFormatter
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from mingle.utils.visualisations.configuration import LatexSizes


def _draw_leaf_boundaries(ax: Axes, leaves: list[Node]) -> None:
    pos = 0
    for leaf in leaves[:-1]:
        pos += len(leaf.idx)
        boundary = pos - 0.5
        ax.axhline(boundary, color="black", linewidth=0.6)
        ax.axvline(boundary, color="black", linewidth=0.6)


def _draw_hierarchical_boundaries(
    ax: Axes, node: Node, start: int = 0, linewidth: float = 1.2
) -> None:
    if node.is_leaf():
        return
    boundary = start + len(node.left.idx)
    ax.axhline(boundary - 0.5, color="black", linewidth=linewidth)
    ax.axvline(boundary - 0.5, color="black", linewidth=linewidth)
    child_lw = max(0.4, linewidth * 0.75)
    _draw_hierarchical_boundaries(ax, node.left, start=start, linewidth=child_lw)
    _draw_hierarchical_boundaries(ax, node.right, start=boundary, linewidth=child_lw)


def plot_adjacency_blocks(
    ax: Axes,
    root: Node,
    W: npt.NDArray[np.floating],
    show_leaf_boundaries: bool = True,
    show_hierarchical_boundaries: bool = True,
) -> AxesImage:
    """Plot a reordered adjacency matrix with optional cluster boundaries."""
    order, leaves = root.get_leaves()

    W_ord = W[np.ix_(order, order)]
    vmax = float(np.nanpercentile(np.abs(W_ord), 99))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    im = ax.imshow(W_ord, aspect="equal", cmap="Reds",
                   vmin=0, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Weight")

    if show_leaf_boundaries: _draw_leaf_boundaries(ax, leaves)
    if show_hierarchical_boundaries: _draw_hierarchical_boundaries(ax, root)

    ax.set_xticks([]); ax.set_yticks([])

    return im


def plot_adjacency_leaf_blocks(
    ax: Axes,
    root: Node,
    W: npt.NDArray[np.floating],
) -> AxesImage:
    """Plot with leaf-level boundaries only."""
    return plot_adjacency_blocks(
        ax, root, W,
        show_leaf_boundaries=True,
        show_hierarchical_boundaries=False,
    )


def plot_adjacency_hierarchy_blocks(
    ax: Axes,
    root: Node,
    W: npt.NDArray[np.floating],
) -> AxesImage:
    """Plot with hierarchical boundaries only."""
    return plot_adjacency_blocks(
        ax, root, W,
        show_leaf_boundaries=False,
        show_hierarchical_boundaries=True,
    )


def plot_adjacency_from_history(
    history,
    decay,
    prices_df,
    cut_alg,
    M=5,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot the learned graph W per window as a reordered adjacency heatmap.

    Cluster blocks come from the cut tree of the first window's induced
    covariance. Inside each block, assets are seriated on the degree-normalised
    W averaged over all windows, so block-diagonal structure is visible. One
    ordering and one set of block boundaries are shared by every panel: only
    the W values vary.
    """
    N = len(history) // M
    if len(history) % M:
        N += 1

    fig, axes = plt.subplots(N, M, figsize=(LatexSizes.DOUBLE_COLUMN, LatexSizes.DOUBLE_COLUMN * N / M))
    axes = axes.flatten()

    # Cluster blocks from the first window only.
    first = history[0]
    first_cov = _induced_covariance(first, prices_df, decay)
    first_root, _ = cut_alg._get_tree(first_cov)
    first_leaves = first_root._collect_leaves()

    leaf_order = np.concatenate([np.asarray(leaf.idx) for leaf in first_leaves]).astype(int)
    boundaries = np.cumsum([len(leaf.idx) for leaf in first_leaves])[:-1].tolist()

    # Seriate within blocks on the average graph, so every panel shares one ordering.
    W_norm = [_degree_normalise(data.W)[np.ix_(leaf_order, leaf_order)] for data in history]
    within = _seriate_within_blocks(np.mean(W_norm, axis=0), boundaries)
    W_norm = [W[np.ix_(within, within)] for W in W_norm]

    shared_vmax: list[float] = []
    for W_ord in W_norm:
        v = float(np.nanpercentile(np.abs(W_ord), 99))
        if np.isfinite(v) and v > 0:
            shared_vmax.append(v)
    vmax = float(np.median(shared_vmax)) if shared_vmax else 1.0

    for i, (data, W_ord) in enumerate(zip(history, W_norm)):

        axes[i].imshow(
            W_ord, aspect="equal", cmap="Reds",
            vmin=0, vmax=vmax, interpolation="nearest",
        )
        # for bd in boundaries:
        #     axes[i].axhline(bd - 0.5, color="black", lw=0.3, alpha=0.6)
        #     axes[i].axvline(bd - 0.5, color="black", lw=0.3, alpha=0.6)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(str(data.start_test)[:10], fontsize=8, fontweight="bold")
        if i % M == 0:
            axes[i].annotate(
                "Rebalancing on",
                xy=(0, 1), xycoords="axes fraction",
                xytext=(-5, 5), textcoords="offset points",
                ha="right", va="bottom",
                fontsize=8,
            )

    for j in range(len(history), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def _degree_normalise(W: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """D^{-1/2} W D^{-1/2}, with isolated nodes left untouched."""
    deg = W.sum(axis=1)
    deg[deg == 0] = 1.0
    return W / np.outer(np.sqrt(deg), np.sqrt(deg))


def _seriate_within_blocks(
    W: npt.NDArray[np.floating], boundaries: list[int]
) -> npt.NDArray[np.intp]:
    """
    Permutation keeping every block [boundaries[i], boundaries[i+1]) contiguous,
    and ordering assets inside each block by average-linkage clustering on W
    (optimal leaf ordering), so strongly connected assets end up adjacent.
    """
    edges = [0, *sorted(b for b in set(boundaries) if 0 < b < len(W)), len(W)]
    order = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.arange(lo, hi)
        if len(idx) > 2:
            S = W[np.ix_(idx, idx)]
            S = (S + S.T) / 2
            D = S.max() - S
            np.fill_diagonal(D, 0.0)
            Z = linkage(squareform(D, checks=False), method="average", optimal_ordering=True)
            idx = idx[leaves_list(Z)]
        order.extend(idx)
    return np.asarray(order)


def within_sector_order(
    W_admms: list[npt.NDArray[np.floating]], sector_boundaries: list[int]
) -> npt.NDArray[np.intp]:
    """
    One within-sector permutation shared by several windows: seriation of the
    degree-normalised learnt graph averaged over `W_admms`. Pass the result as
    `perm` to every plot_adjacency_sector_boundaries call so panels stay
    comparable.
    """
    W_mean = np.mean([_degree_normalise(W) for W in W_admms], axis=0)
    return _seriate_within_blocks(W_mean, sector_boundaries)


def plot_adjacency_sector_boundaries(
    W_admm: npt.NDArray[np.floating],
    W_corr: npt.NDArray[np.floating],
    sector_boundaries: list[int],
    label: Optional[str] = None,
    perm: Optional[npt.NDArray[np.intp]] = None,
    axes=None,
    show_titles: bool = True,
    show_xlabel: bool = True,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    if perm is None:
        perm = within_sector_order([W_admm], sector_boundaries)
    W_admm = W_admm[np.ix_(perm, perm)]
    W_corr = W_corr[np.ix_(perm, perm)]

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        fig.subplots_adjust(left=0.10, right=0.92, wspace=0.25)
    else:
        fig = axes[0].get_figure()

    col_titles = ["Learnt graph topology", "Correlation graph"]

    for col, W_plot in enumerate([_degree_normalise(W_admm), W_corr]):

        ax = axes[col]
        nz = W_plot[W_plot > 0]

        if col == 1:
            vmin = 0.0
            vmax = 1.0
        else:
            vmin = 0.0
            vmax = float(np.percentile(nz, 90)) if len(nz) else 1.0

        im = ax.imshow(
            np.clip(W_plot, 0, vmax),
            aspect="auto",
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        cbar.formatter = FuncFormatter(lambda x, pos: np.format_float_positional(
            float(f"{x:.1g}"),
            trim="-"
        ))
        cbar.update_ticks()
        if col == 1:
            cbar.set_label("Edge weight")

        for bd in sector_boundaries:
            ax.axhline(bd - 0.5, color="black", lw=0.7, alpha=0.6)
            ax.axvline(bd - 0.5, color="black", lw=0.7, alpha=0.6)

        ax.set_xticks([])
        ax.set_yticks([])
        if show_titles: ax.set_title(col_titles[col])
        if show_xlabel: ax.set_xlabel("Assets")

    axes[0].set_ylabel(f"{label}\n\nAssets" if label else "Assets")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_sector_comparison(
    windows: list[LoadedWindow],
    prices_df: pd.DataFrame,
    sectors: pd.Series,
    vix: pd.Series,
    regimes: pd.Series,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Learnt graph vs density-matched |corr| graph, GICS-sector ordered, for
    the calmest and most stressed rebalancing windows.
    """
    def mean_vix(w: LoadedWindow) -> float:
        return float(vix.loc[w.info.train_start:w.info.train_end].mean())

    scored = [w for w in windows if np.isfinite(mean_vix(w))]
    chosen = [min(scored, key=mean_vix), max(scored, key=mean_vix)]

    order = np.argsort(sectors.to_numpy(), kind="stable")
    sorted_sectors = sectors.to_numpy()[order]
    sector_boundaries = [i for i in range(1, len(sorted_sectors)) if sorted_sectors[i] != sorted_sectors[i - 1]]

    matrices, labels = [], []
    for w in chosen:
        X_log = log_returns(prices_df, w.info.train_start, w.info.train_end)
        W_corr = density_match_correlation(build_abs_correlation_graph(X_log), w.W)
        matrices.append((w.W[np.ix_(order, order)], W_corr[np.ix_(order, order)]))
        labels.append(f"{regimes.asof(w.info.test_start)} market regime")

    # Maintain within-sector ordering for every row
    perm = within_sector_order([W_admm for W_admm, _ in matrices], sector_boundaries)

    fig, axes = plt.subplots(len(matrices), 2, figsize=(7, 6))
    fig.subplots_adjust(left=0.10, right=0.92, hspace=0.1, wspace=0.25)
    for row, ((W_admm, W_corr), label) in enumerate(zip(matrices, labels)):
        plot_adjacency_sector_boundaries(
            W_admm, W_corr, sector_boundaries, label=label, perm=perm, axes=axes[row],
            show_titles=row == 0, show_xlabel=row == len(matrices) - 1,
        )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def _induced_covariance(data, prices_df: pd.DataFrame, decay: float) -> npt.NDArray[np.floating]:
    """ADMM-induced covariance Σ = B Σ_F Bᵀ + Ψ for a rebalancing window."""
    B, F = data.B, data.F
    X_log = log_returns(prices_df, data.start_train, data.end_train)
    omega = get_observations_decay(decay, F.shape[1])
    _, _, cov = get_representation_cov(X_log, B, F, omega)
    return cov
