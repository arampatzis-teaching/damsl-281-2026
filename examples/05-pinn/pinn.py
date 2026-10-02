"""The pieces of a physics-informed neural network, in PyTorch.

A network stands for the solution, a residual says what the equation asks of
it, and a loss adds the residual to whatever else the solution must satisfy.
Nothing here is specific to one equation: the residual is passed in as a
function.
"""

import numpy as np
import torch
from torch import nn


def device():
    """The fastest device available: Metal, CUDA, or the processor."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLP(nn.Module):
    """A fully connected network with tanh activations and a linear output."""

    def __init__(self, sizes):
        super().__init__()
        layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(n_in, n_out), nn.Tanh()]
        self.net = nn.Sequential(*layers[:-1])   # drop the last activation

    def forward(self, x):
        return self.net(x)


class Constrained(nn.Module):
    """A network that satisfies the initial condition and the period exactly.

    The solution is written as u(x,t) = u0(x) + (1 - exp(-t)) net(x,t), which
    equals u0 at t = 0 whatever the weights, and net is fed sin(pi x) and
    cos(pi x) instead of x, which makes it 2-periodic whatever the weights.
    """

    def __init__(self, u0, sizes):
        super().__init__()
        self.u0 = u0
        self.net = MLP([3] + list(sizes) + [1])

    def forward(self, X):
        x, t = X[:, :1], X[:, 1:]
        features = torch.cat([torch.sin(np.pi * x), torch.cos(np.pi * x), t], 1)
        return self.u0(x) + (1 - torch.exp(-t)) * self.net(features)


def d(y, x, order=1):
    """The derivative of y with respect to x, taken order times."""
    for _ in range(order):
        (y,) = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)
    return y


def points(bounds, n, device=None):
    """n points drawn uniformly from a box, ready to be differentiated."""
    lo = torch.tensor([b[0] for b in bounds], device=device)
    hi = torch.tensor([b[1] for b in bounds], device=device)
    x = lo + (hi - lo) * torch.rand(n, len(bounds), device=device)
    return x.requires_grad_(True)


def allen_cahn_residual(net, X, diffusion=2e-5):
    """u_t - D u_xx - u + u^3 at the points X, whose columns are x and t."""
    u = net(X)
    u_x, u_t = d(u, X).split(1, dim=1)
    u_xx = d(u_x, X)[:, :1]
    return u_t - diffusion * u_xx - u + u**3


def train(loss_fn, parameters, steps, lr=1e-3, anneal=True):
    """Minimise loss_fn with Adam and return the loss at every step."""
    parameters = list(parameters)
    opt = torch.optim.Adam(parameters, lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    history = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if anneal:
            schedule.step()
        history.append(loss.item())
    return history
