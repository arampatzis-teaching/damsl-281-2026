"""POD-Galerkin for the Allen-Cahn system du/dt = D L u + u - u^3.

The reduced right-hand side V^T f(V z) is assembled once from the basis V:
a matrix for the linear part and a tensor of order four for the cube, so
that evaluating it costs nothing that grows with the dimension d.
"""

import numpy as np


def reduced_operators(V, L, D):
    """A = D V^T L V and T[e,a,b,c] = sum_i V[i,e] V[i,a] V[i,b] V[i,c]."""
    A = D * (V.T @ (L @ V))
    T = np.einsum("ie,ia,ib,ic->eabc", V, V, V, V, optimize=True)
    return A, T


def reduced_rhs(A, T):
    """The function z -> A z + z - T(z, z, z) of the reduced system."""
    def rhs(t, z):
        return A @ z + z - np.einsum("eabc,a,b,c->e", T, z, z, z, optimize=True)
    return rhs
