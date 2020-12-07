from typing import Any, Callable, Optional, Tuple

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

from flax.linen import Module, compact

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?


def _absolute_dims(rank, dims):
    return tuple([rank + dim if dim < 0 else dim for dim in dims])


class BatchNorm(Module):
    track_running_stats: bool = True
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @compact
    def __call__(self, x, training: bool = False):
        x = jnp.asarray(x, jnp.float32)
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = _absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

        # we detect if we're in initialization via empty variable tree.
        initializing = not self.has_variable('batch_stats', 'mean')

        ra_mean = self.variable('batch_stats', 'mean',
                                lambda s: jnp.zeros(s, jnp.float32),
                                reduced_feature_shape)
        ra_var = self.variable('batch_stats', 'var',
                               lambda s: jnp.ones(s, jnp.float32),
                               reduced_feature_shape)

        if not self.track_running_stats or not training:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
            mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
            if self.axis_name is not None and not initializing:
                concatenated_mean = jnp.concatenate([mean, mean2])
                mean, mean2 = jnp.split(
                    lax.pmean(
                        concatenated_mean,
                        axis_name=self.axis_name,
                        axis_index_groups=self.axis_index_groups), 2)
            var = mean2 - lax.square(mean)

            if not initializing:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        y = x - mean.reshape(feature_shape)
        mul = lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = self.param('scale',
                               self.scale_init,
                               reduced_feature_shape).reshape(feature_shape)
            mul = mul * scale
        y = y * mul
        if self.use_bias:
            bias = self.param('bias',
                              self.bias_init,
                              reduced_feature_shape).reshape(feature_shape)
            y = y + bias
        return jnp.asarray(y, self.dtype)
