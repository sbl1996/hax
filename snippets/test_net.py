from jax import numpy as jnp, random
import flax
from hax.models.cifar.preactresnet import ResNet

model = ResNet(depth=10, k=2, dropout=0.3, num_classes=100)

rng = random.PRNGKey(0)
rng, dropout_rng = random.split(rng, 2)

x = random.normal(rng, (1, 32, 32, 3))

variables = model.init({"params": rng, "dropout": dropout_rng}, x)
y, new_vars = model.apply(variables, x, True, mutable="batch_stats")
t1 = new_vars['batch_stats']['PreActDownBlock_0']['BatchNorm_0']['var'].mean()
y, new_vars = model.apply(new_vars, x, False, mutable="batch_stats")
t2 = new_vars['batch_stats']['PreActDownBlock_0']['BatchNorm_0']['var'].mean()
y, new_vars = model.apply(new_vars, x, True, mutable="batch_stats")
t3 = new_vars['batch_stats']['PreActDownBlock_0']['BatchNorm_0']['var'].mean()


def to_dict(x):
    if isinstance(x, flax.core.FrozenDict):
        return {k: to_dict(v) for k, v in x.items()}
    else:
        return x
