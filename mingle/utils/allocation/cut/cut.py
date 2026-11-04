import numpy as np
import logging
from typing import Callable
from heapq import heapify, heappush, heappop
from mingle.utils.allocation.cut.node import Node
from abc import abstractmethod, ABC
from collections import deque

CutAlgorithmLogger = logging.getLogger(__name__)

class CutAlgorithm(ABC):

    def __init__(self, splits_count: int=15, min_in_cluster: int=5):
        self.splits_count = splits_count
        self.min_in_cluster = min_in_cluster

    @abstractmethod
    def solve_eigenproblem(self, W):
        pass

    @abstractmethod
    def get_cluster_quantity(self, W, idx):
        pass

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
            vals_diff = np.diff(vals[start:])
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
    
    def get_weights(self, W, get_multiplier: Callable, eigen_lower_threshold=1e-8, **kwargs):

        N = W.shape[0]

        root, cut_eigenvals = self._get_tree(W, eigen_lower_threshold=eigen_lower_threshold)
        queue = deque([root])
        depth = 0

        weights = np.zeros(N)

        while queue:

            level_cnt = len(queue)

            for _ in range(level_cnt):

                node = queue.popleft()

                if not node.left and not node.right:
                    idx = node.idx
                    multiplier = get_multiplier(idx, **kwargs)
                    weights[idx] = multiplier / (2**depth * multiplier.sum())

                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)

            depth += 1

        return weights / weights.sum(), cut_eigenvals
