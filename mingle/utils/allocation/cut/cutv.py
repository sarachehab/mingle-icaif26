import numpy as np
from mingle.utils.graph import laplacian_from_W, degrees_from_W
from mingle.utils.allocation.cut.cut import CutAlgorithm, CutAlgorithmLogger
from heapq import heapify, heappush, heappop
from mingle.utils.allocation.cut.node import Node

class CutVAlgorithm(CutAlgorithm):

    def solve_eigenproblem(self, W: np.ndarray):

        # Determine out-degrees of nodes
        degrees = degrees_from_W(W)

        # Determine diffusion model of graph
        L = laplacian_from_W(W)

        # Determine D^-0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.abs(degrees)))
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0

        # Solve generalised eigenproblem D^-0.5 L D^-0.5 x = λ x 
        L_mod = D_inv_sqrt @ L @ D_inv_sqrt
        return np.linalg.eigh(L_mod)
 
    def get_cluster_quantity(self, W: np.ndarray, idx: np.ndarray) -> float:
        W_sub = W[np.ix_(idx, idx)]
        return np.abs(W_sub).sum()

    def _get_tree(self, W, eigen_lower_threshold=1e-8):

        N = W.shape[0]
        idx = np.arange(N)
        root = Node(idx)
        cut_eigenvals = []

        priority_queue = [(-N, -N, id(root), idx, root)]
        heapify(priority_queue)

        for _ in range(self.splits_count):
            if not priority_queue:
                break

            _, _, _, idx, parent = heappop(priority_queue)

            # Solve generalised eigenproblem
            W_sub = W[np.ix_(idx, idx)]
            vals, vecs = self.solve_eigenproblem(W_sub)

            # Skip trivial zero eigenvalue
            start = np.searchsorted(vals, eigen_lower_threshold, side="right")
            vals_diff = np.diff(vals[start-1:])
            if vals_diff.size == 0: continue

            # Find the largest eigengap
            i_max = start + np.argmax(vals_diff)
            vec_max = vecs[:, i_max]
            lambda_cut = vals[i_max]

            # Split by sign
            right_idx = idx[vec_max < 0]
            left_idx = idx[vec_max >= 0]

            # Skip if split is trivial
            nb_eigenvals = W_sub.shape[0]
            if len(right_idx) == 0 or len(left_idx) == 0:
                CutAlgorithmLogger.debug(f"Skipping split on eigenvalue {i_max} / {nb_eigenvals}")
                continue

            CutAlgorithmLogger.debug(f"Splitting on eigenvalue {i_max} / {nb_eigenvals}")
            cut_eigenvals.append((i_max, nb_eigenvals))

            # Save nodes
            right = Node(right_idx)
            left = Node(left_idx)
            parent.right = right
            parent.left = left

            right_qty = self.get_cluster_quantity(W, right_idx) / lambda_cut
            left_qty = self.get_cluster_quantity(W, left_idx) / lambda_cut

            if len(right_idx) > self.min_in_cluster:
                heappush(priority_queue, (
                    -right_qty, -len(right_idx), id(right), right_idx, right))
            if len(left_idx) > self.min_in_cluster:
                heappush(priority_queue, (
                    -left_qty, -len(left_idx), id(left), left_idx, left))
        
        return root, cut_eigenvals
