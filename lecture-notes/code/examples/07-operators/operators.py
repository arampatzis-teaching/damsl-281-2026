"""Two neural operators for functions on the periodic interval (-1, 1).

A DeepONet reads the input function at fixed sensor points and returns a
function that can be evaluated at any point. A Fourier neural operator reads
the input on a uniform grid of any size and returns the output on the same
grid. Both are written for scalar functions of one variable.
"""

import math

import torch
from torch import nn


def grid(n):
    """The n points x_i = -1 + 2i/n, i = 0, ..., n-1, of the periodic interval."""
    return -1 + 2 * torch.arange(n) / n


def mlp(sizes, activation=nn.Tanh):
    """A fully connected network with the given layer sizes and a linear output."""
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(n_in, n_out), activation()]
    return nn.Sequential(*layers[:-1])


class DeepONet(nn.Module):
    """G(u)(x) = sum_k beta_k(u(x^s_1), ..., u(x^s_m)) tau_k(x) + b_0.

    The branch network beta reads the input at m sensors, the trunk network
    tau reads (sin pi x, cos pi x), so every output is 2-periodic in x.
    """

    def __init__(self, m, n_tau=64, width=128, depth=3):
        super().__init__()
        self.branch = mlp([m] + [width] * depth + [n_tau])
        self.trunk = mlp([2] + [width] * depth + [n_tau])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, x):
        """u: (batch, m) sensor values; x: (d,) query points. Returns (batch, d)."""
        beta = self.branch(u)
        tau = self.trunk(torch.stack([torch.sin(math.pi * x),
                                      torch.cos(math.pi * x)], -1))
        return beta @ tau.T + self.bias


class FourierLayer(nn.Module):
    """v -> A v + b + F^{-1}(R . F v), with R acting on the lowest `modes` modes.

    With norm="forward" rfft returns (1/n) sum_i v_i exp(-2 pi sqrt(-1) k i/n).
    The coefficients of the notes refer to the points x_i = -1 + 2i/n and
    differ from these by the factor (-1)^k, which the inverse transform undoes,
    so the output is the same. The factor 1/n makes the coefficients
    approximate those of the underlying function whatever n is.
    """

    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / width                         # a small spectral part at first
        self.R_re = nn.Parameter(scale * torch.randn(width, width, modes))
        self.R_im = nn.Parameter(scale * torch.randn(width, width, modes))
        self.A = nn.Conv1d(width, width, 1)            # A v + b, point by point

    def forward(self, v):
        """v: (batch, width, n). Returns (batch, width, n)."""
        n = v.shape[-1]
        v_hat = torch.fft.rfft(v, norm="forward")[..., : self.modes]
        R = torch.complex(self.R_re, self.R_im)
        out = torch.einsum("bim,iom->bom", v_hat, R)   # R[:, :, k] is R_k transposed
        spectral = torch.fft.irfft(out, n=n, norm="forward")  # modes -k: conjugates
        return self.A(v) + spectral


class FNO(nn.Module):
    """A Fourier neural operator: lift, Fourier layers with GELU, project.

    The input is u on the grid x_i = -1 + 2i/n for any n, and the output is on
    the same grid. No layer sees the position x, so shifting the input by a
    whole number of grid points shifts the output by the same amount.
    """

    def __init__(self, width=32, modes=16, layers=4):
        super().__init__()
        self.lift = nn.Conv1d(1, width, 1)
        self.layers = nn.ModuleList(FourierLayer(width, modes) for _ in range(layers))
        self.project = nn.Sequential(nn.Conv1d(width, 128, 1), nn.GELU(),
                                     nn.Conv1d(128, 1, 1))

    def forward(self, u):
        """u: (batch, n) values on the uniform grid. Returns (batch, n)."""
        v = self.lift(u.unsqueeze(1))
        for ell, layer in enumerate(self.layers):
            v = layer(v)
            if ell < len(self.layers) - 1:
                v = nn.functional.gelu(v)
        return self.project(v).squeeze(1)


def relative_error(pred, true):
    """The mean over the batch of ||pred - true|| / ||true|| on the grid."""
    return ((pred - true).norm(dim=-1) / true.norm(dim=-1)).mean()


def pooled_error(pred, true):
    """(sum ||pred - true||^2 / sum ||true||^2)^(1/2) over the whole batch."""
    return ((pred - true).norm() / true.norm())
