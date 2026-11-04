import numpy as np
from mingle.utils.graph import laplacian_from_W
from mingle.utils.allocation.cut.cut import CutAlgorithm

class CutNAlgorithm(CutAlgorithm):

    def solve_eigenproblem(self, W):
        L = laplacian_from_W(W)
        return np.linalg.eigh(L)
    
    def get_cluster_quantity(self, W, idx):
        return len(idx)
