"""A neural ordinary differential equation and the two ways of training it.

The state obeys du/dt = f(u, w) with f a network. Training needs the gradient
of a loss with respect to w, and this module has both ways of getting it: by
differentiating through the steps of the solver, and by the adjoint equations
of the chapter on adjoint sensitivity, solved backwards in time.
"""

import torch
from torch import nn


class Field(nn.Module):
    """A small tanh network standing for the right-hand side of the equation."""

    def __init__(self, dim, width=64, depth=2):
        super().__init__()
        sizes = [dim] + [width] * depth + [dim]
        layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(n_in, n_out), nn.Tanh()]
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, t, u):
        return self.net(u)


def rk4_step(f, t, u, dt):
    """One step of the classical fourth-order Runge-Kutta method."""
    s1 = f(t, u)
    s2 = f(t + dt / 2, u + dt / 2 * s1)
    s3 = f(t + dt / 2, u + dt / 2 * s2)
    s4 = f(t + dt, u + dt * s3)
    return u + dt / 6 * (s1 + 2 * s2 + 2 * s3 + s4)


def odeint(f, u0, t):
    """Solve du/dt = f(t,u) at the times t, returning the whole trajectory.

    The steps stay in the graph, so differentiating the result differentiates
    through the solver.
    """
    us = [u0]
    for t0, t1 in zip(t[:-1], t[1:]):
        us.append(rk4_step(f, t0, us[-1], t1 - t0))
    return torch.stack(us)


def _add(state, direction, factor):
    """state + factor * direction, for a state held as a list of tensors."""
    return [x + factor * d for x, d in zip(state, direction)]


class _AdjointSolve(torch.autograd.Function):
    """odeint with the backward pass replaced by the adjoint equations.

    Nothing of the forward solve is kept except its final state. The backward
    pass integrates the augmented system of the adjoint chapter,

        du/dt = f(u),
        dl/dt = -(df/du)^T l,
        dgamma/dt = -(df/dw)^T l,

    backwards from t = T, adding the jump dL/du(t_k) to l at every output
    time, and returns gamma as the gradient with respect to the weights.
    """

    @staticmethod
    def forward(ctx, f, u0, t, *parameters):
        with torch.no_grad():
            us = odeint(f, u0, t)
        ctx.f = f
        ctx.save_for_backward(t, us[-1])
        return us

    @staticmethod
    def backward(ctx, grad_us):
        f = ctx.f
        t, u_final = ctx.saved_tensors
        parameters = tuple(f.parameters())

        def rhs(s, state):
            """The right-hand side of the augmented system, all three parts."""
            u, lam = state[0], state[1]
            with torch.enable_grad():
                u = u.detach().requires_grad_(True)
                out = f(s, u)
                vjp = torch.autograd.grad(out, (u,) + parameters, -lam,
                                          allow_unused=True)
            grads = [torch.zeros_like(p) if v is None else v
                     for p, v in zip(parameters, vjp[1:])]
            return [out.detach(), vjp[0], *grads]

        state = [u_final, torch.zeros_like(u_final)]
        state += [torch.zeros_like(p) for p in parameters]

        for k in range(len(t) - 1, 0, -1):
            state[1] = state[1] + grad_us[k]              # the jump at t_k
            dt = -(t[k] - t[k - 1])                       # backwards in time
            s1 = rhs(t[k], state)
            s2 = rhs(t[k] + dt / 2, _add(state, s1, dt / 2))
            s3 = rhs(t[k] + dt / 2, _add(state, s2, dt / 2))
            s4 = rhs(t[k] + dt, _add(state, s3, dt))
            step = [(a + 2 * b + 2 * c + d) / 6
                    for a, b, c, d in zip(s1, s2, s3, s4)]
            state = _add(state, step, dt)
        state[1] = state[1] + grad_us[0]
        return (None, state[1], None, *state[2:])


def adjoint_odeint(f, u0, t):
    """Like odeint, but the gradient comes from the adjoint equations."""
    return _AdjointSolve.apply(f, u0, t, *f.parameters())
