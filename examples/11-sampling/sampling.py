"""Sampling a density pi on R^p known up to a constant.

Every sampler takes the logarithm of the unnormalised target, log_pi, so the
normalising constant is never needed. The Markov chain samplers return the
whole chain, one row per step, and the fraction of accepted proposals.
"""

import numpy as np


def snis(log_pi, sample_q, log_q, f, n, rng):
    """Self-normalised importance sampling estimate of E_pi[f].

    Returns the estimate and the effective sample size of the weights.
    """
    theta = sample_q(n, rng)
    log_w = log_pi(theta) - log_q(theta)
    w = np.exp(log_w - log_w.max())                    # scale cancels below
    w /= w.sum()
    return np.sum(w * f(theta)), 1.0 / np.sum(w**2)


def metropolis(log_pi, theta0, n, h, C, rng):
    """Random-walk Metropolis with proposals theta + h C^(1/2) eps."""
    L = np.linalg.cholesky(C)
    theta = np.array(theta0, dtype=float)
    lp = log_pi(theta)
    chain, accepted = np.empty((n, theta.size)), 0
    for k in range(n):
        prop = theta + h * L @ rng.standard_normal(theta.size)
        lp_prop = log_pi(prop)
        if np.log(rng.random()) < lp_prop - lp:                        #<1>
            theta, lp, accepted = prop, lp_prop, accepted + 1
        chain[k] = theta
    return chain, accepted / n


def mala(log_pi_grad, theta0, n, dt, C, rng):
    """Metropolis-adjusted Langevin with preconditioner C and step dt.

    log_pi_grad(theta) returns log pi(theta) and its gradient.
    """
    L = np.linalg.cholesky(C)
    Cinv = np.linalg.inv(C)

    def log_q(to, frm, g_frm):              # log density of the proposal, up to a constant
        r = to - frm - 0.5 * dt * C @ g_frm
        return -0.5 * r @ Cinv @ r / dt

    theta = np.array(theta0, dtype=float)
    lp, g = log_pi_grad(theta)
    chain, accepted = np.empty((n, theta.size)), 0
    for k in range(n):
        prop = (theta + 0.5 * dt * C @ g
                + np.sqrt(dt) * L @ rng.standard_normal(theta.size))   #<1>
        lp_prop, g_prop = log_pi_grad(prop)
        log_alpha = (lp_prop + log_q(theta, prop, g_prop)
                     - lp - log_q(prop, theta, g))                     #<2>
        if np.log(rng.random()) < log_alpha:
            theta, lp, g, accepted = prop, lp_prop, g_prop, accepted + 1
        chain[k] = theta
    return chain, accepted / n


def ula(grad_log_pi, theta0, n, dt, rng):
    """The unadjusted Langevin algorithm: Euler-Maruyama, no correction."""
    theta = np.array(theta0, dtype=float)
    chain = np.empty((n, theta.size))
    for k in range(n):
        theta = (theta + 0.5 * dt * grad_log_pi(theta)
                 + np.sqrt(dt) * rng.standard_normal(theta.size))
        chain[k] = theta
    return chain


def autocorrelation(x):
    """Sample autocorrelation of a scalar chain at every lag, by FFT."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n] / n
    return acov / acov[0]


def iat(x, c=5.0):
    """Integrated autocorrelation time, with Sokal's automatic window."""
    rho = autocorrelation(x)
    tau = 1 + 2 * np.cumsum(rho[1:])
    for m in range(1, len(tau)):
        if m >= c * tau[m - 1]:
            return tau[m - 1]
    return tau[-1]


def ess(x):
    """Effective sample size of a scalar chain, n / tau."""
    return len(x) / iat(x)
