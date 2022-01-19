import time
import inspect
from functools import partial
from toolz import curry
from concurrent.futures import thread

from typing import Any, Sequence, Type

import numpy as np
import tensorflow as tf

import jax
from jax import lax
import jax.numpy as jnp

from flax import linen as nn
from flax import jax_utils
from flax.training import train_state
from flax.training import common_utils
import optax

from hhutil.io import time_now

from hanser.transform import to_tensor, normalize, random_crop
from hanser.datasets.cifar import make_cifar100_dataset

from hax.optim.sgd import sgd
from hax.optim.lr_schedule import cosine_lr

broadcast = jax_utils.replicate

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
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training=False):
        norm = nn.BatchNorm(momentum=self.momentum, epsilon=self.epsilon, dtype=self.dtype)
        return norm(x, use_running_average=not training)


class ReLU(nn.Module):
    def __call__(self, x):
        return nn.relu(x)


def log_metrics(stage, metrics):
    end_at = time_now()
    log_str = "%s %s - " % (end_at, stage)
    metric_logs = []
    for k, v in metrics.items():
        metric_logs.append("%s: %.4f" % (k, v))
    log_str += ", ".join(metric_logs)
    print(log_str)


def prepare_tf_data(xs):
    local_device_count = jax.local_device_count()

    def _prepare(x):
        x = x._numpy()
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def get_iter(dataset):
    it = map(prepare_tf_data, dataset)
    return it


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
        return ReLU()(identity + x)


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


def Linear(in_channels, out_channels, dtype=jnp.float32):
    return nn.Dense(out_channels, use_bias=True, dtype=dtype)


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


def cross_entropy_loss(logits, labels):
    return optax.softmax_cross_entropy(logits=logits, labels=labels)


def compute_metrics(logits, labels):
    return {
        'total': jnp.int32(logits.shape[0]),
        'loss': cross_entropy_loss(logits, labels).sum(),
        'acc': jnp.sum(jnp.argmax(logits, -1) == jnp.argmax(labels, -1)),
    }


def train_step(state, batch, prev_metrics):

    images, labels = batch
    images = images.astype(jnp.bfloat16)

    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            images, training=True, mutable=['batch_stats'])
        logits = logits.astype(jnp.float32)
        per_example_loss = cross_entropy_loss(logits, labels)
        loss = jnp.mean(per_example_loss)
        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name='batch')
    logits, new_model_state = aux[1]

    metrics = compute_metrics(logits, labels)
    metrics = jax.tree_multimap(jnp.add, prev_metrics, metrics)
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats'])
    return new_state, metrics


def allreduce_metrics(metrics):
    return lax.psum(metrics, axis_name='batch')


def make_train_function(
    input_shape, num_classes, batch_size, steps_per_loop):

    n = jax.device_count()

    def train_loop_cond(args):
        step = args[-1]
        return step // steps_per_loop != 1

    def train_loop_body(args):
        state, metrics, token, step = args
        batch, token = lax.infeed(token, shape=(
            jax.ShapedArray((batch_size // n, *input_shape), jnp.float32),
            jax.ShapedArray((batch_size // n, num_classes), jnp.float32)))
        state, metrics = train_step(state, batch, metrics)
        step += 1
        return state, metrics, token, step

    def train_loop(state, metrics, step):
        token = lax.create_token()
        state, metrics, _token, step = lax.while_loop(
            train_loop_cond,
            train_loop_body,
            (state, metrics, token, step))
        metrics = allreduce_metrics(metrics)
        return state, metrics, step

    train_functon = jax.pmap(train_loop, axis_name='batch')
    return train_functon


def train_epoch(train_function, state, train_it, steps_per_epoch):
    host_step, device_step = 0, broadcast(0)
    empty_metrics = broadcast({'total': 0, 'loss': 0., 'acc': 0})
    state, metrics, device_step = train_function(state, empty_metrics, device_step)

    infeed_pool = thread.ThreadPoolExecutor(jax.local_device_count(), 'infeed')
    local_devices = jax.local_devices()

    while host_step < steps_per_epoch:
        while infeed_pool._work_queue.qsize() > 100:
            time.sleep(0.01)
        batch = next(train_it)
        for i, device in enumerate(local_devices):
            infeed_pool.submit(partial(device.transfer_to_infeed, jax.tree_map(lambda x: x[i], batch)))
        host_step += 1

    metrics = jax.tree_map(lambda x: jax.device_get(x[0]), metrics)
    total = metrics.pop('total')
    metrics = jax.tree_map(lambda x: x / total, metrics)
    state = sync_batch_stats(state)
    return state, metrics


def eval_step(state, batch):
    images, labels = batch
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        images, training=False)
    logits = logits.astype(jnp.float32)
    metrics = compute_metrics(logits, labels)
    metrics = allreduce_metrics(metrics)
    return metrics


def eval_epoch(eval_step, state, test_it, test_steps):
    eval_metrics = []
    for i in range(test_steps):
        metrics = eval_step(state, next(test_it))
        eval_metrics.append(metrics)
    metrics = common_utils.get_metrics(eval_metrics)
    metrics = jax.tree_map(lambda x: x.sum(), metrics)
    total = metrics.pop('total')
    metrics = jax.tree_map(lambda x: x / total, metrics)
    return metrics


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')

def sync_batch_stats(state):
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


class TrainState(train_state.TrainState):
    batch_stats: Any


@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)

    image, label = to_tensor(image, label)
    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

    label = tf.one_hot(label, 100)
    return image, label


batch_size = 128
eval_batch_size = 2048

ds_train, ds_test, steps_per_epoch, test_steps = make_cifar100_dataset(
    batch_size, eval_batch_size, transform)

rng = jax.random.PRNGKey(0)

rng, init_rng = jax.random.split(rng)

input_shape = (32, 32, 3)
num_classes = 100
model = ResNet(depth=110, block=Bottleneck, num_classes=num_classes, dtype=jnp.bfloat16)
variables = jax.jit(model.init, device=jax.devices(backend="cpu")[0])(init_rng, jnp.ones([1, *input_shape]))
params, batch_stats = variables['params'], variables['batch_stats']

base_lr = 0.1
epochs = 200
lr_schedule = cosine_lr(base_lr, steps_per_epoch, epochs, warmup_epoch=5)
tx = sgd(lr_schedule, momentum=0.9, nesterov=True, weight_decay=5e-4)

state = TrainState.create(
    apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
state = jax_utils.replicate(state)


train_function = make_train_function(
    input_shape, num_classes, batch_size, steps_per_epoch)
p_eval_step = jax.pmap(eval_step, axis_name='batch')

train_it = get_iter(ds_train)
test_it = get_iter(ds_test)

for epoch in range(epochs):
    print("Epoch %d/%d" % (epoch + 1, epochs))
    state, metrics = train_epoch(train_function, state, train_it, steps_per_epoch)
    log_metrics('train', metrics)

    metrics = eval_epoch(p_eval_step, state, test_it, test_steps)
    log_metrics('valid', metrics)