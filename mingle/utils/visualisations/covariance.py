from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from mingle.utils.admm import get_observations_decay, get_representation_cov
from mingle.utils.metrics.covariance import (
    raw_sample_cov,
    induced_asset_cov,
    condition_number,
)
from mingle.utils.transforms import log_returns
from mingle.utils.runner.loader.experiment import LoadedWindow


def plot_cov_condition_evolution(
    windows: list[LoadedWindow],
    prices_df,
    decay: float,
    ax=None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Time series: condition number of the raw sample covariance vs the
    Representation-induced covariance Σ = BΣ_F Bᵀ + Ψ over rolling windows.

    Args:
        windows : list of LoadedWindow (mingle.utils.runner.loader.experiment),
        prices_df : pd.DataFrame
        decay : float
        ax : optional matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig = ax.get_figure()

    dates, raw_conds, ind_conds = [], [], []

    for w in windows:
        B, F = w.B, w.F
        X_log = log_returns(prices_df, w.info.train_start, w.info.train_end)
        omega = get_observations_decay(decay, F.shape[1])
        F_cov, _, _ = get_representation_cov(X_log, B, F, omega)

        Sigma_raw = raw_sample_cov(X_log, omega)
        Sigma_ind = induced_asset_cov(B, F_cov, X_log, F, omega)

        dates.append(w.info.test_start)
        raw_conds.append(condition_number(Sigma_raw))
        ind_conds.append(condition_number(Sigma_ind))

    dates = np.asarray(dates)

    ax.plot(dates, raw_conds, linewidth=0.8, label=r"Raw covariance $\mathbf X \mathbf X^\top$")
    ax.plot(dates, ind_conds, linewidth=0.8, label=r"Induced covariance $\mathbf B \Sigma_\mathbf{F} \mathbf{B}^\top + \Psi$")
    ax.set_ylabel("Condition number")
    ax.set_xlabel("Date")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
