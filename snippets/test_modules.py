from typing import Tuple

import jax
from jax import numpy as jnp, random
from flax import linen as nn
from hax.models.bn import BatchNorm
from hax.models.dropout import Dropout


class FC(nn.Module):
    features: Tuple[int] = (16, 4)

    def setup(self):
        self.dense1 = nn.Dense(self.features[0])
        self.bn1 = BatchNorm()
        self.dropout1 = Dropout(0.5)
        self.dense2 = nn.Dense(self.features[1])

    def __call__(self, x, training):
        x = self.dense1(x)
        x = self.bn1(x, training)
        x = nn.relu(x)
        x = self.dropout1(x, training)
        x = self.dense2(x)
        return x


rng = random.PRNGKey(0)
rng, dropout_rng = random.split(rng, 2)

x = random.normal(rng, (2, 2))

m = FC()
variables = m.init({"params": rng, "dropout": dropout_rng}, x, True)
y, new_vars = m.apply(variables, x, True, rngs={"params": rng, "dropout": dropout_rng}, mutable="batch_stats")
y, new_vars = m.apply(new_vars, x, False, rngs={"params": rng, "dropout": dropout_rng}, mutable="batch_stats")

t1 = new_vars['batch_stats']['bn1']['mean'].mean()