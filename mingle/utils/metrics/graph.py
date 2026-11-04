import numpy as np

from mingle.utils.graph import laplacian_from_W


def fiedler(W: np.ndarray) -> float:
    """
    Measures algebraic connectivity.
        Higher = more strongly connected network, and harder to separate into disconnected components.
    """
    L = laplacian_from_W(W)
    eigvals = np.linalg.eigvalsh(L)
    return float(eigvals[1])
