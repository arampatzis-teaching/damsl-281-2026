"""A Bayesian network with independent normal factors on its weights.

Every weight w_j has the factor N(m_j, s_j^2) with s_j = log(1 + exp(rho_j)),
and the prior is N(0, s0^2) for every weight. A layer either samples its
weights, or samples its output directly (the local reparametrisation).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesLinear(nn.Module):
    """An affine layer h = V a + b with independent normal weights."""

    def __init__(self, n_in, n_out, s0=1.0):
        super().__init__()
        self.s0 = s0
        self.m_w = nn.Parameter(torch.randn(n_out, n_in) / math.sqrt(n_in))
        self.m_b = nn.Parameter(torch.zeros(n_out))
        self.rho_w = nn.Parameter(torch.full((n_out, n_in), -5.0))
        self.rho_b = nn.Parameter(torch.full((n_out,), -5.0))

    def forward(self, a, local=True):
        s_w, s_b = F.softplus(self.rho_w), F.softplus(self.rho_b)
        if local:                                                     #<1>
            mean = a @ self.m_w.T + self.m_b
            var = (a**2) @ (s_w**2).T + s_b**2
            return mean + var.sqrt() * torch.randn_like(mean)
        V = self.m_w + s_w * torch.randn_like(s_w)                    #<2>
        b = self.m_b + s_b * torch.randn_like(s_b)
        return a @ V.T + b

    def kl(self):
        """Divergence of the factors from the prior, in closed form."""
        out = 0.0
        for m, rho in [(self.m_w, self.rho_w), (self.m_b, self.rho_b)]:
            s = F.softplus(rho)
            out = out + 0.5 * ((s**2 + m**2) / self.s0**2 - 1
                               - 2 * torch.log(s / self.s0)).sum()    #<3>
        return out


class BayesNet(nn.Module):
    """One hidden layer of n_h units with the activation max(u, 0)."""

    def __init__(self, n_h=50, s0=1.0):
        super().__init__()
        self.l1, self.l2 = BayesLinear(1, n_h, s0), BayesLinear(n_h, 1, s0)

    def forward(self, x, local=True):
        return self.l2(torch.relu(self.l1(x, local)), local)

    def kl(self):
        return self.l1.kl() + self.l2.kl()


def neg_bound(model, x, y, n_total, sigma, local=True):
    """Minus the estimate of the bound from a subset (x, y) of n_total pairs."""
    pred = model(x, local)
    loglik = (-0.5 * (y - pred)**2 / sigma**2
              - math.log(sigma) - 0.5 * math.log(2 * math.pi)).sum()
    return -(n_total / len(x) * loglik - model.kl())                  #<4>
