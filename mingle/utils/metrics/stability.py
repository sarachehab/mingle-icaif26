import numpy as np
import pandas as pd


def build_abs_correlation_graph(X: np.ndarray) -> np.ndarray:
    """
    Compute |corr(X)| with the diagonal zeroed.

    Args:
        X: returns matrix (p, T)

    Returns
        C_abs: absolute-correlation matrix, diagonal = 0 (p, p)
    """
    C = np.corrcoef(X)
    np.fill_diagonal(C, 0.0)
    return np.abs(C)


def density_match_correlation(C_abs: np.ndarray, W_B: np.ndarray) -> np.ndarray:
    """
    Construct a correlation graph with the same edge density as W_B.

    Steps:
      1. Count non-zero upper-triangular edges in W_B.
      2. Keep the same number of largest upper-triangular entries in C_abs.
      3. Zero out all other entries.
      4. Symmetrise.

    Args:
        C_abs: absolute-correlation matrix (diagonal already zeroed) (p,p)
        W_B: exposure adjacency matrix (non-negative, symmetric) (p,p)

    Returns
        W_corr: density-matched correlation graph (symmetric, non-negative) (p, p)
    """
    n = W_B.shape[0]
    upper_rows, upper_cols = np.triu_indices(n, k=1)

    n_edges = int((W_B[upper_rows, upper_cols] > 0).sum())
    vals = C_abs[upper_rows, upper_cols]

    W_corr = np.zeros_like(C_abs)
    if n_edges == 0:
        return W_corr

    # Threshold: the n_edges-th largest value
    sorted_vals = np.sort(vals)[::-1]
    threshold = sorted_vals[min(n_edges, len(sorted_vals) - 1)]

    kept = np.where(vals >= threshold, vals, 0.0)
    W_corr[upper_rows, upper_cols] = kept
    W_corr = W_corr + W_corr.T   # symmetrise (diagonal stays 0)
    return W_corr


def gics_sector_purity(labels: np.ndarray, gics: pd.Series) -> float:
    """
    Fraction of intra-cluster asset pairs that share the same GICS sector.

    Only considers assets that have a non-NaN GICS label, allowing partial
    coverage. Returns NaN if fewer than 2 assets have GICS labels.

    Interpretation: higher purity -> clusters group economically similar assets.
    """
    mask = gics.notna().values
    if mask.sum() < 2:
        return float("nan")

    labels_m = labels[mask]
    gics_m = gics.values[mask]
    n = len(labels_m)

    same_cluster = labels_m.reshape(-1, 1) == labels_m.reshape(1, -1)
    same_sector = gics_m.reshape(-1, 1) == gics_m.reshape(1, -1)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    total_intra = (same_cluster & upper).sum()
    if total_intra == 0:
        return 0.0
    return float((same_cluster & same_sector & upper).sum() / total_intra)

