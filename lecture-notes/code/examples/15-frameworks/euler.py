"""One step of the forward Euler method, written twice.

The functions are pure: they read their arguments, return a value, and change
nothing else. The class keeps the state of the integration inside the object,
which is the style most courses teach first.
"""

import jax


def euler_step(u, dt, f):
    """Return the state after one step of size dt."""
    return u + dt * f(u)


def euler_run(u0, dt, steps, f):
    """Return the state after the given number of steps."""
    u = u0
    for _ in range(steps):
        u = euler_step(u, dt, f)
    return u


def trajectory(u0, dt, steps, f):
    """Return the list of states, without changing anything outside."""
    u, states = u0, [u0]
    for _ in range(steps):
        u = euler_step(u, dt, f)
        states.append(u)
    return states


class EulerSolver:
    """The same integration, with the state kept inside the object."""

    def __init__(self, f, dt):
        self.f = f
        self.dt = dt
        self.u = None

    def start(self, u0):
        self.u = u0

    def step(self):
        self.u = euler_step(self.u, self.dt, self.f)
        return self.u


def euler_scan(u0, dt, steps, f):
    """The same loop, handed to jax.lax.scan so that it compiles once."""

    def step(u, _):
        return u + dt * f(u), None

    u, _ = jax.lax.scan(step, u0, None, length=steps)
    return u
