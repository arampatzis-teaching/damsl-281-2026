"""A small network in Flax, used by the session of Section 15.5."""

import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    """One hidden layer, with the parameters held by the object."""

    def __init__(self, din, dhidden, dout, rngs):
        self.l1 = nnx.Linear(din, dhidden, rngs=rngs)
        self.l2 = nnx.Linear(dhidden, dout, rngs=rngs)

    def __call__(self, x):
        return self.l2(jnp.tanh(self.l1(x)))
