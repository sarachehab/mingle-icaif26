from __future__ import annotations

from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

_ADMM_COLS = [
    ("Primal residual",  ("r", "eps_pri_")),
    ("Dual residual",    ("s", "eps_dual_")),
    ("Penalty $\\rho$",  ("rho_",)),
]
_ADMM_VARS = ["B", "W", "d"]


def plot_admm_convergence(
    history,
    title: str = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Diagnostic grid for one ADMM window: primal residuals, dual residuals, and penalty ρ."""
    n_vars = len(_ADMM_VARS)
    n_cols = len(_ADMM_COLS)

    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(n_vars, n_cols, figsize=(5 * n_cols, 3 * n_vars), sharex=True)

    for j, var in enumerate(_ADMM_VARS):
        for i, (col_title, attrs) in enumerate(_ADMM_COLS):
            ax = axes[j][i]

            for attr in attrs:
                attr_name = f"{attr}{var}"

                if attr.startswith("rho"):
                    label = "ADMM penalty"
                    color = cmap(2)
                    linestyle = "-"
                elif attr.startswith(("eps_pri_", "eps_dual_")):
                    label = "Residual"
                    color = cmap(1)
                    linestyle = "-"
                else:
                    label = "Tolerance"
                    color = cmap(0)
                    linestyle = "--"

                values = [getattr(step, attr_name) for step in history]
                ax.plot(values, label=label, color=color, linestyle=linestyle)
            ax.set_yscale("log")
            if j == 0: ax.set_title(col_title)
            if j == n_vars - 1: ax.set_xlabel("Epoch")
        axes[j][0].set_ylabel(f"Variable {var}")

    if title: fig.suptitle(title)

    legend_handles = [
        Line2D([], [], color=cmap(0), linestyle="--", label="Residual"),
        Line2D([], [], color=cmap(1), linestyle="-", label="Tolerance"),
        Line2D([], [], color=cmap(2), linestyle="-", label="ADMM penalty"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )

    if save_path is not None: fig.savefig(save_path, bbox_inches="tight")
    return fig
