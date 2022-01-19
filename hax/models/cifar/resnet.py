from typing import Sequence

from functools import partial

import jax.numpy as jnp
from flax import linen as nn
from hax.models.layers import DType, ModuleDef, Conv2d, ReLU, Linear, Act


class Bottleneck(nn.Module):
    in_channels: int
    channels: ModuleDef
    stride: int
    expansion: int = 4
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, x, training=False):
        out_channels = self.channels * self.expansion

        identity = x
        x = Conv2d(self.in_channels, self.channels, 1,
                   norm='def', act='def', dtype=self.dtype)(x, training=training)
        x = Conv2d(self.channels, self.channels, 3, stride=self.stride,
                   norm='def', act='def', dtype=self.dtype)(x, training=training)
        x = Conv2d(self.channels, out_channels, 1,
                   norm='def', dtype=self.dtype)(x, training=training)

        if identity.shape != x.shape:
            identity = Conv2d(self.in_channels, out_channels, 1, stride=self.stride,
                              norm='def', dtype=self.dtype)(identity, training=training)
        return Act()(identity + x)


class ResNet(nn.Module):
    depth: int
    block: ModuleDef
    num_classes: int = 10
    channels: Sequence[int] = (16, 16, 32, 64)
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, x, training=False):
        block = partial(self.block, dtype=self.dtype)
        layers = [(self.depth - 2) // 9] * 3
        stem_channels, *channels = self.channels
        x = Conv2d(3, stem_channels, kernel_size=3,
                   norm='def', act='def', dtype=self.dtype)(x, training=training)

        c_in = stem_channels

        strides = [1, 2, 2]
        for i, (c, n, s) in enumerate(zip(channels, layers, strides)):
            x = block(c_in, c, stride=s)(x, training=training)
            c_in = c * self.block.expansion
            for i in range(1, n):
                x = block(c_in, c, stride=1)(x, training=training)

        x = jnp.mean(x, axis=(1, 2))
        x = Linear(c_in, self.num_classes, dtype=self.dtype)(x)
        return x


def resnet110(**kwargs):
    return ResNet(depth=110, block=Bottleneck, **kwargs)