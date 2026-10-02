"""Two ways of advancing a state that is a large array.

Both functions are pure in the sense that matters to the caller: neither
touches an array the caller still needs. The first asks for new memory at
every call, the second writes into a buffer it was given.
"""

import numpy as np


def step_new(u, dt):
    """Return the new state as a new array."""
    return u - dt * u


def step_into(u, dt, out):
    """Write the new state into out and return it."""
    np.multiply(u, 1.0 - dt, out=out)
    return out
