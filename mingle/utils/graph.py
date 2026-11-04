import numpy as np
from scipy.linalg import eigh

def degrees_from_W(W):
    return np.sum(W, axis=1)

def laplacian_from_W(W):
    return np.diag(degrees_from_W(W)) - W

def reorder_adjacency_matrix_by_cluster(W, labels):
    labels = np.asarray(labels)
    order = np.argsort(labels, kind="stable")
    return W[np.ix_(order, order)], order
