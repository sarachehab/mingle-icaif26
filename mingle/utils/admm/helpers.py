import numpy as np

def pairwise_sq_dists_rows(B):
    """
    Compute the pairwise squared Euclidean distances between the rows of B.

        C_{i,j} = || B_i - B_j ||^2_2

    Args:
        B: Matrix whose rows are the vectors B_i, (N, M)
    
    Returns:
        C: Matrix whose entries are the pairwise Euclidean distances, (M, M)
    """
    # Determine norm of individual rows
    col_dist = np.sum(B * B, axis=1, keepdims=True)

    # Determine norm of row-difference: ||B_i - B_j||^2 = ||B_i||^2 + ||B_j||^2 - B_i^T B_j
    C = col_dist + col_dist.T - 2.0 * (B @ B.T)

    # Ensure numerical safety
    C = np.maximum(C, 0.0)
    np.fill_diagonal(C, 0.0)

    return C

def l1_proximal_operator(M, gamma):
    """
    Applies element-wise proximal operator of the l1 norm.

        Prox_{γ, |.|} (m) = argmin_x  γ ||x||_1 + (1/2) ||x - m||_F^2

    The solution is given by element-wise soft-thresholding:

        soft(m) = sign(m) * max(m - γ, 0)

    Args:
        M: Input matrix to be thresholded, (n, m)
        gamma: Nonnegative threshold weight

    Returns:
        X: Soft-thresholded matrix, (m, n)
    """
    return np.sign(M) * np.maximum(np.abs(M) - gamma, 0.0)

def euclidean_projection(M):
    """
    Projects M into a Euclidean subspace by applying:
        - symmetrisation
        - non-negativity
        - zero diagonal

    Args:
        M: Input matrix (m, m)

    Returns:
        X: Projected matrix (m, m)
    """
    # Project into the symmetric matrix subspace
    symmetric = 0.5 * (M + M.T)

    # Project into the non-negative orthant
    non_neg = np.maximum(symmetric, 0.0)

    # Project onto zero-diagonal matrices
    np.fill_diagonal(non_neg, 0.0)
    return non_neg

def get_observations_decay(decay: float, T: int) -> np.ndarray:
    """
    Calculates observations decay matrix.

    Ensures more recent observations are more closely fitted by the ADMM loop.

    Args:
        decay: exponential decay factor
        T: time horizon

    Returns:
        omega: Weights of observations
    """
    # Assign same importance to observations for no decay
    if decay == 1.0: return np.ones(T)

    # Assign exponentially decreasing importance for observations
    omega = np.array([decay ** (T - 1 - t) for t in range(T)])

    # Normalise weights to sum to T
    omega  = omega / omega.sum() * T
    
    return omega

def get_representation_cov_nolowrank(Y: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Weighted sample covariance of unconstrained representation Y.

    Replaces get_representation_cov in the no-low-rank ablation where X ≈ Y
    directly (no BF decomposition), so the full (p, p) covariance is computed
    from Y rather than assembled from factor_cov + idiosyncratic_var.

    Args:
        Y: Unconstrained representation (p, T)
        omega: Recency-decay weights (T,), summing to T

    Returns:
        cov: Weighted sample covariance (p, p)
    """
    w = omega / omega.sum()
    Y_mean = (Y * w).sum(axis=1, keepdims=True)
    Y_centered = Y - Y_mean
    return (Y_centered * w) @ Y_centered.T


def get_representation_cov(X, B, F, omega):
    """
    Calculate the covariance of the factor model.
        || (X - BF) Ω^0.5 ||_F^2 = ∑ Ω_t || X_t - BF_t ||^2

    This is equivalent to the quantity
        Cov(X) = Cov(BF) + Var(X-BF)

    where:
        Cov(F)  = 1 / T * ∑ Ω_t F_t F_t^T
        Var(E)  = Var((X - BF) Ω^0.5)

    Args:
        X: Fitted time series (p, T)
        B: Exposures (p, k)
        F: Factors (l, T)
        Omega: Weights, sum up to T (T, )

    Returns:
        factor_covar: Cov(F) (k, k)
        idiosyncratic_var: Var(E) (p, )

    """
    w = omega / omega.sum()

    # Weighted covariance of factors
    F_mean = (F * w).sum(axis=1, keepdims=True)
    F_centered = F - F_mean
    factor_cov = (F_centered * w) @ F_centered.T

    # Weighted residual variances
    E = X - B @ F
    E_mean = (E * w).sum(axis=1, keepdims=True)
    E_centered = E - E_mean
    res_var = (E_centered**2 * w).sum(axis=1)

    # Full covariance
    cov = B @ factor_cov @ B.T + np.diag(res_var)

    return factor_cov, res_var, cov
