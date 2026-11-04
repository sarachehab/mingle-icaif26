import numpy as np

class Node:
    
    def __init__(self, idx: np.ndarray):
        self.idx: np.ndarray = idx
        self.left: Node = None
        self.right: Node = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None
    
    def _collect_leaves(self):

        if self.is_leaf(): return [self]
        
        leaves: list[Node] = []

        if self.left: leaves.extend(self.left._collect_leaves())
        if self.right: leaves.extend(self.right._collect_leaves())

        return leaves
    
    def get_leaves(self):
        leaves = self._collect_leaves()
        order = np.concatenate([leaf.idx for leaf in leaves])
        return order, leaves
