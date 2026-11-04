import numpy as np


def raw_sample_cov(X: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Weighted sample covariance of X (n_assets × T) under decay weights omega (T,).

    Cov = (X * omega) @ X.T / sum(omega)
    """
    w = omega / omega.sum()
    X_w = X * w
    return X_w @ X.T


def induced_asset_cov(
    B: np.ndarray,
    F_cov: np.ndarray,
    X: np.ndarray,
    F: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """
    ADMM-induced full asset covariance: Σ = B Σ_F Bᵀ + Ψ

    Ψ is the diagonal matrix of weighted residual variances:
        residual = X - B F
        Ψ_ii = weighted variance of residual row i
    """
    residual = X - B @ F
    w = omega / omega.sum()
    psi_diag = (residual ** 2 * w).sum(axis=1)
    Psi = np.diag(psi_diag)
    return B @ F_cov @ B.T + Psi


def condition_number(Sigma: np.ndarray) -> float:
    # Extract eigenvalies from positive definite matrix Sigma
    eigvals = np.linalg.eigvalsh(Sigma)
    eigvals = np.maximum(eigvals, 0.0)

    # Clipping to avoid division by zero
    lam_min = eigvals[eigvals > 0].min() if (eigvals > 0).any() else 1e-12

    # Obtain condition number 
    return float(eigvals.max() / lam_min)
