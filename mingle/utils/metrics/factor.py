import numpy as np

from mingle.utils.admm import get_representation_cov


def factor_cov_corr(
    X: np.ndarray,
    B: np.ndarray,
    F: np.ndarray,
    omega: np.ndarray,
) -> tuple:
    """
    Compute the factor covariance and correlation matrices.

    Args:
        X: log-return matrix (n_assets, T)
        B: factor loading matrix (n_assets, k)
        F: factor matrix (k, T)
        omega: per-observation decay weights (T,)

    Returns
        F_cov: factor covariance matrix (k,k)
        F_corr: factor correlation matrix (k,k)
    """
    F_cov, _, _ = get_representation_cov(X, B, F, omega)
    std = np.sqrt(np.diag(F_cov))
    F_corr = F_cov / np.outer(std, std)
    return F_cov, F_corr


def reconstruction_loss(X: np.ndarray, B: np.ndarray, F: np.ndarray, omega: np.ndarray) -> float:
    """
    Weighted Frobenius reconstruction loss
        ||( X - B F ) * sqrt(omega)||_F.
    """
    return float(np.linalg.norm((X - B @ F) * np.sqrt(omega)))


def frac_off_diagonal(F_corr: np.ndarray) -> float:
    """
    Percentage of total absolute correlation mass sitting off the diagonal.
    A value near 0 means factors are nearly orthogonal.
    """
    off_diag = F_corr - np.diag(np.diag(F_corr))
    return float(100 * np.abs(off_diag).sum() / np.abs(F_corr).sum())


def hhi(F_cov: np.ndarray, exclude_f: int = 0) -> float:
    """
    Herfindahl-Hirschman Index of factor variance concentration.

        1/k: Equally distributed:
        1: Concentrated in one factor.
    """
    variances = np.diag(F_cov[exclude_f:, exclude_f:])
    shares = variances / variances.sum()
    return float(np.sum(shares ** 2))
