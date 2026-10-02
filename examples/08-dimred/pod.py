"""Proper orthogonal decomposition of a matrix of snapshots.

The columns of Y are the data vectors. The POD basis of rank r is orthonormal
in the inner product <u, v>_W = u^T W v and spans the best r-dimensional
subspace for the data in the norm of W.
"""

import numpy as np
from scipy.linalg import solve_triangular


def pod(Y, r, W=None):
    """The POD basis V (d x r) of the columns of Y, and all singular values.

    With W = M^T M the Cholesky factorisation, the basis is M^{-1} times the
    first r left singular vectors of M Y, so that V^T W V = I.
    """
    if W is None:
        Vbar, s, _ = np.linalg.svd(Y, full_matrices=False)
        return Vbar[:, :r], s
    M = np.linalg.cholesky(W).T                       # W = M^T M, M upper triangular
    Vbar, s, _ = np.linalg.svd(M @ Y, full_matrices=False)
    return solve_triangular(M, Vbar[:, :r]), s


def floor(s):
    """E_q = (sum_{i>q} s_i^2 / sum_i s_i^2)^(1/2) for q = 0, 1, ..., len(s)."""
    tail = np.concatenate([np.cumsum((s**2)[::-1])[::-1], [0.0]])
    return np.sqrt(tail / tail[0])
