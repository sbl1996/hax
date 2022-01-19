import inspect
from typing import Sequence, Type, Any
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any
DType = Any

class Sequential(nn.Module):

    layers: Sequence[Type[nn.Module]]

    def __call__(self, x, training=False):
        for layer in self.layers:
            sig = inspect.signature(layer)
            if 'training' in sig.parameters:
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


class BatchNorm(nn.Module):

    momentum: float = 0.9
    epsilon: float = 1e-5
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, x, training=False):
        norm = nn.BatchNorm(momentum=self.momentum, epsilon=self.epsilon, dtype=self.dtype)
        return norm(x, use_running_average=not training)


class ReLU(nn.Module):
    def __call__(self, x):
        return nn.relu(x)


def Conv2d(in_channels, out_channel, kernel_size, stride=1, padding='SAME', bias=None,
           norm=None, act=None, dtype=jnp.float32):
    if bias is None:
        bias = norm is None
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    conv = nn.Conv(out_channel, kernel_size, strides=stride, padding=padding, use_bias=bias, dtype=dtype)
    layers = [conv]
    if norm is not None:
        layers.append(Norm(out_channel, dtype=dtype))
    if act is not None:
        layers.append(ReLU())
    return Sequential(layers)


def Norm(channels, dtype=jnp.float32):
    return BatchNorm(momentum=0.9, epsilon=1e-5, dtype=dtype)


def Act(act):
    return ReLU()


def Linear(in_channels, out_channels, dtype=jnp.float32):
    return nn.Dense(out_channels, use_bias=True, dtype=dtype)
