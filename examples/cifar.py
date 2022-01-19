import time
from functools import partial
from toolz import curry
from concurrent.futures import thread

from typing import Any

import tensorflow as tf

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

from flax import jax_utils
from flax.training import train_state
from flax.training import common_utils

from hhutil.io import time_now

from hanser.transform import to_tensor, normalize, random_crop
from hanser.datasets.cifar import make_cifar100_dataset

from hax.optim import sgd, cosine_lr
from hax.nn.loss import cross_entropy
from hax.models.cifar.resnet import resnet110
from hax.models.cifar.preactresnet import wrn_28_10

broadcast = jax_utils.replicate


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


def compute_metrics(logits, labels):
    return {
        'total': jnp.int32(logits.shape[0]),
        'loss': cross_entropy(logits, labels).sum(),
        'acc': jnp.sum(jnp.argmax(logits, -1) == jnp.argmax(labels, -1)),
    }


def train_step(state, batch, prev_metrics, dropout_rng=None):

    images, labels = batch
    images = images.astype(jnp.bfloat16)

    dropout_rng, new_dropout_rng = random.split(dropout_rng)

    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats}, images,
            training=True, mutable=['batch_stats'], rngs={'dropout': dropout_rng})
        logits = logits.astype(jnp.float32)
        per_example_loss = cross_entropy(logits, labels)
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
    return new_state, metrics, new_dropout_rng


def allreduce_metrics(metrics):
    return lax.psum(metrics, axis_name='batch')


def make_train_function(
    input_shape, num_classes, batch_size, steps_per_loop):

    n = jax.device_count()

    def train_loop_cond(args):
        step = args[-1]
        return step // steps_per_loop != 1

    def train_loop_body(args):
        state, metrics, dropout_rng, token, step = args
        batch, token = lax.infeed(token, shape=(
            jax.ShapedArray((batch_size // n, *input_shape), jnp.float32),
            jax.ShapedArray((batch_size // n, num_classes), jnp.float32)))
        state, metrics, dropout_rng = train_step(state, batch, metrics, dropout_rng)
        step += 1
        return state, metrics, dropout_rng, token, step

    def train_loop(state, metrics, dropout_rng, step):
        token = lax.create_token()
        state, metrics, dropout_rng, _token, step = lax.while_loop(
            train_loop_cond,
            train_loop_body,
            (state, metrics, dropout_rng, token, step))
        metrics = allreduce_metrics(metrics)
        return state, metrics, dropout_rng, step

    train_functon = jax.pmap(train_loop, axis_name='batch')
    return train_functon


def train_epoch(train_function, state, train_it, steps_per_epoch, dropout_rng):
    host_step, device_step = 0, broadcast(0)
    empty_metrics = broadcast({'total': 0, 'loss': 0., 'acc': 0})
    state, metrics, dropout_rng, device_step = train_function(
        state, empty_metrics, dropout_rng, device_step)

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
    return state, metrics, dropout_rng


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
model = wrn_28_10(dropout=0.3, num_classes=num_classes, dtype=jnp.bfloat16)
variables = jax.jit(model.init, device=jax.devices(backend="cpu")[0])(init_rng, jnp.ones([1, *input_shape]))
params, batch_stats = variables['params'], variables['batch_stats']

base_lr = 0.1
epochs = 200
lr_schedule = cosine_lr(base_lr, steps_per_epoch, epochs, warmup_epoch=5)
tx = sgd(lr_schedule, momentum=0.9, nesterov=True, weight_decay=5e-4)

state = TrainState.create(
    apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
state = broadcast(state)


train_function = make_train_function(
    input_shape, num_classes, batch_size, steps_per_epoch)
p_eval_step = jax.pmap(eval_step, axis_name='batch')

train_it = get_iter(ds_train)
test_it = get_iter(ds_test)

dropout_rng = random.split(rng, jax.local_device_count())

for epoch in range(epochs):
    print("Epoch %d/%d" % (epoch + 1, epochs))
    state, metrics, dropout_rng = train_epoch(train_function, state, train_it, steps_per_epoch, dropout_rng)
    log_metrics('train', metrics)

    metrics = eval_epoch(p_eval_step, state, test_it, test_steps)
    log_metrics('valid', metrics)


import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
