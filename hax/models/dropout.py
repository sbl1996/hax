from flax.linen import Module, compact
from jax import lax, random, numpy as jnp


class Dropout(Module):
    rate: float

    @compact
    def __call__(self, inputs, training=False, rng=None):
        if self.rate == 0.:
            return inputs
        keep_prob = 1. - self.rate
        if not training:
            return inputs
        else:
            if rng is None:
                rng = self.make_rng('dropout')
            mask = random.bernoulli(rng, p=keep_prob, shape=inputs.shape)
            return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))
