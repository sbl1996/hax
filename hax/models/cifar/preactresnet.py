from typing import Sequence
from functools import partial

import jax.numpy as jnp
from flax import linen as nn
from hax.models.layers import NormAct, Conv2d, Linear, DType, ModuleDef, Dropout


class BasicBlock(nn.Module):
    in_channels: int
    channels: int
    stride: int
    dropout: float
    expansion: int = 1
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, x, training):
        out_channels = self.channels * self.expansion

        shortcut = x

        x = NormAct(dtype=self.dtype)(x, training)

        if self.in_channels != out_channels or self.stride == 2:
            shortcut = Conv2d(self.in_channels, out_channels, 1, stride=self.stride, dtype=self.dtype)(x)

        x = Conv2d(self.in_channels, out_channels, 3, stride=self.stride,
                   norm='def', act='def', dtype=self.dtype)(x, training)

        if self.dropout:
            x = Dropout(self.dropout)(x, training)

        x = Conv2d(out_channels, out_channels, 3, dtype=self.dtype)(x)
        return shortcut + x


class ResNet(nn.Module):
    depth: int
    block: ModuleDef
    num_classes: int = 10
    channels: Sequence[int] = (16, 16, 32, 64)
    dropout: float = 0.3
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, x, training=False):
        block = partial(self.block, dropout=self.dropout, dtype=self.dtype)
        layers = [(self.depth - 4) // 6] * 3
        stem_channels, *channels = self.channels
        x = Conv2d(3, stem_channels, kernel_size=3, dtype=self.dtype)(x)

        c_in = stem_channels

        strides = [1, 2, 2]
        for i, (c, n, s) in enumerate(zip(channels, layers, strides)):
            x = block(c_in, c, stride=s)(x, training=training)
            c_in = c * self.block.expansion
            for i in range(1, n):
                x = block(c_in, c, stride=1)(x, training=training)

        x = NormAct(dtype=self.dtype)(x, training)
        x = jnp.mean(x, axis=(1, 2))
        x = Linear(c_in, self.num_classes, dtype=self.dtype)(x)
        return x


def wrn_28_10(**kwargs):
    channels = [16, 16, 32, 64]
    k = 10
    channels = (channels[0],) + tuple(c * k for c in channels[1:])
    return ResNet(depth=28, block=BasicBlock, channels=channels, **kwargs)