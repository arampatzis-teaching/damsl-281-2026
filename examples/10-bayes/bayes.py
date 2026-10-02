"""Bayesian inversion: the linear Gaussian model in closed form, and the
Allen-Cahn forward map with its sensitivities for the Laplace approximation.

The forward map takes the parameters (a, D) of du/dt = D L u + a (u - u^3)
to the solution at chosen grid points and times. With sens=True it also
returns its Jacobian, from the forward sensitivity equations of Chapter 4,
ds/dt = J_f(u) s + df/dtheta, solved together with the state.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, diags


def gaussian_posterior(G, y, Sigma, m0, C0):
    """Posterior N(m, C) and log evidence for y = G theta + eps.

    The noise is eps ~ N(0, Sigma) and the prior theta ~ N(m0, C0).
    """
    C = np.linalg.inv(G.T @ np.linalg.solve(Sigma, G) + np.linalg.inv(C0))  #<1>
    m = C @ (G.T @ np.linalg.solve(Sigma, y) + np.linalg.solve(C0, m0))
    S = Sigma + G @ C0 @ G.T                                            #<2>
    r = y - G @ m0
    log_evidence = -0.5 * (np.linalg.slogdet(2 * np.pi * S)[1]
                           + r @ np.linalg.solve(S, r))
    return m, C, log_evidence


def laplacian(d):
    """Second-difference matrix on d periodic points of (-1, 1)."""
    dx = 2 / d
    return diags([1.0, -2.0, 1.0, 1.0, 1.0], [-1, 0, 1, d - 1, -(d - 1)],
                 shape=(d, d)).tocsr() / dx**2


def forward(a, D, points, times, d=1024, sens=False, rtol=1e-8, atol=1e-10):
    """The solution at grid indices `points` and `times`, as one vector.

    The vector lists all points at the first time, then at the second, and
    so on. With sens=True, also its Jacobian with respect to (a, D).
    """
    x = -1 + 2 * np.arange(d) / d
    L = laplacian(d)
    u0 = x**2 * np.cos(np.pi * x)

    def jac_f(u):
        return D * L + a * diags(1 - 3 * u**2)

    if not sens:
        sol = solve_ivp(lambda t, u: D * (L @ u) + a * (u - u**3),
                        (0, times[-1]), u0, t_eval=times, method="BDF",
                        jac=lambda t, u: jac_f(u), rtol=rtol, atol=atol)
        return sol.y[points].T.ravel()

    def rhs(t, v):
        u, sa, sD = v[:d], v[d:2 * d], v[2 * d:]
        J = jac_f(u)
        return np.concatenate([D * (L @ u) + a * (u - u**3),
                               J @ sa + (u - u**3),                 #<1>
                               J @ sD + L @ u])

    def jac(t, v):
        u, sa, sD = v[:d], v[d:2 * d], v[2 * d:]
        J = jac_f(u)
        return bmat([[J, None, None],                                  #<2>
                     [diags(1 - 3 * u**2 - 6 * a * u * sa), J, None],
                     [L - diags(6 * a * u * sD), None, J]], format="csr")

    sol = solve_ivp(rhs, (0, times[-1]), np.concatenate([u0, 0 * u0, 0 * u0]),
                    t_eval=times, method="BDF", jac=jac, rtol=rtol, atol=atol)
    u, sa, sD = sol.y[:d], sol.y[d:2 * d], sol.y[2 * d:]
    G = u[points].T.ravel()
    J = np.column_stack([sa[points].T.ravel(), sD[points].T.ravel()])  #<3>
    return G, J
