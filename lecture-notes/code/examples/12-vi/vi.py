"""Gaussian variational inference with the reparametrisation gradient.

The approximating density is N(m, R R^T) with R lower triangular and a
positive diagonal, or diagonal for the mean-field family. The diagonal is
optimised through its logarithm rho, so that it stays positive. The target
enters only through log_p_grad(theta), which returns the logarithm of the
unnormalised target and its gradient.
"""

import numpy as np


def gaussian_kl(m, C, m0, C0):
    """Kullback-Leibler divergence of N(m, C) from N(m0, C0)."""
    p = len(m)
    C0inv = np.linalg.inv(C0)
    dm = m - m0
    return 0.5 * (np.trace(C0inv @ C) + dm @ C0inv @ dm - p
                  + np.linalg.slogdet(C0)[1] - np.linalg.slogdet(C)[1])


def entropy(rho):
    """Entropy of N(m, R R^T) when the diagonal of R is exp(rho)."""
    p = len(rho)
    return np.sum(rho) + 0.5 * p * (1 + np.log(2 * np.pi))


def factor(rho, off, mean_field):
    """The lower-triangular factor R from its log-diagonal and entries below it."""
    p = len(rho)
    R = np.diag(np.exp(rho))
    if not mean_field:
        R[np.tril_indices(p, -1)] = off
    return R


def gaussian_vi(log_p_grad, m0, rho0, steps, rng, S=2, lr=0.1,
                mean_field=False):
    """Maximise the bound over N(m, R R^T) by stochastic gradient ascent.

    Adam with a step size decreasing geometrically from lr to lr / 10, and
    the parameters averaged over the last quarter of the steps. Returns m,
    rho, the entries of R below the diagonal, and the history of the bound
    estimates.
    """
    p = len(m0)
    params = [np.array(m0, float), np.array(rho0, float), np.zeros(p * (p - 1) // 2)]
    first = [np.zeros_like(x) for x in params]
    second = [np.zeros_like(x) for x in params]
    below = np.tril_indices(p, -1)
    history = []
    last = steps // 4
    average = [np.zeros_like(x) for x in params]
    for k in range(steps):
        m, rho, off = params
        R = factor(rho, off, mean_field)
        grads = [np.zeros(p), np.zeros(p), np.zeros_like(off)]
        values = []
        for _ in range(S):
            eps = rng.standard_normal(p)
            lp, g = log_p_grad(m + R @ eps)                           #<1>
            values.append(lp)
            grads[0] += g / S                                          #<2>
            grads[1] += (np.exp(rho) * g * eps + 1) / S                #<3>
            if not mean_field:
                grads[2] += np.outer(g, eps)[below] / S
        history.append(np.mean(values) + entropy(rho))
        step = lr * 0.1 ** (k / steps)
        for i in range(3):                                             #<4>
            first[i] = 0.9 * first[i] + 0.1 * grads[i]
            second[i] = 0.999 * second[i] + 0.001 * grads[i] ** 2
            mhat = first[i] / (1 - 0.9 ** (k + 1))
            vhat = second[i] / (1 - 0.999 ** (k + 1))
            params[i] = params[i] + step * mhat / (np.sqrt(vhat) + 1e-8)
        if k >= steps - last:                                          #<5>
            for i in range(3):
                average[i] += params[i] / last
    return average[0], average[1], average[2], np.array(history)


def elbo(log_p, m, R, n, rng):
    """Monte Carlo estimate of the bound and its standard error, n samples."""
    p = len(m)
    vals = np.array([log_p(m + R @ rng.standard_normal(p)) for _ in range(n)])
    h = np.sum(np.log(np.diag(R))) + 0.5 * p * (1 + np.log(2 * np.pi))
    return vals.mean() + h, vals.std() / np.sqrt(n)
