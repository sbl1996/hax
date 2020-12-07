import jax.numpy as jnp
from flax import linen as nn
from hax.models.layers import Act, Norm, Conv2d, Linear


class PreActDownBlock(nn.Module):
    channels: int
    stride: int
    dropout: float

    @nn.compact
    def __call__(self, x, training):
        x = Norm()(x, training)
        x = Act()(x)
        residual = x
        x = Conv2d(self.channels, 3, self.stride)(x)
        x = Norm()(x, training)
        x = Act()(x)
        if self.dropout:
            x = nn.Dropout(self.dropout)(x, training)
        x = Conv2d(self.channels, 3)(x)

        residual = Conv2d(self.channels, 1, self.stride)(residual)

        return residual + x


class PreActResBlock(nn.Module):
    channels: int
    dropout: float

    @nn.compact
    def __call__(self, x, training):
        residual = x
        x = Norm()(x, training)
        x = Act()(x)
        x = Conv2d(self.channels, 3)(x)

        x = Norm()(x, training)
        x = Act()(x)
        x = Conv2d(self.channels, 3)(x)
        if self.dropout:
            x = nn.Dropout(self.dropout)(x, training)
        return residual + x


class ResNet(nn.Module):
    depth: int
    k: int
    dropout: float = 0.0
    num_classes: int = 10
    stem_channels: int = 16

    @nn.compact
    def __call__(self, x, training: bool = False):
        num_blocks = (self.depth - 4) // 6
        strides = [1, 2, 2]
        channels = [16, 32, 64]

        x = Conv2d(self.stem_channels, 3, 1)(x)
        for s, c in zip(strides, channels):
            x = PreActDownBlock(c * self.k, s, self.dropout)(x, training)
            for j in range(1, num_blocks):
                x = PreActResBlock(c * self.k, self.dropout)(x, training)
        x = Norm()(x, training)
        x = Act()(x)
        x = jnp.mean(x, axis=(1, 2))
        x = Linear(self.num_classes)(x)
        return x
