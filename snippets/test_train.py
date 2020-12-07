import functools
from functools import partial
import time
from typing import Any

from flax.training import common_utils
from toolz import curry

import flax
from flax import jax_utils
from flax import optim

import jax
from jax import lax
from jax import random

import jax.numpy as jnp
from hanser.transform import random_crop, normalize, to_tensor

import tensorflow as tf

from hanser.datasets.cifar import make_cifar10_dataset
from hax.models.cifar.preactresnet import ResNet
from hax.nn.loss import cross_entropy
from hax.optim.lr_schedule import cosine_lr


@flax.struct.dataclass
class TrainState:
    step: int
    optimizer: optim.Optimizer
    model_state: Any


def initialized(key, shape, model):
    input_shape = (1,) + shape

    @jax.jit
    def init(*args):
        return model.init(*args)

    key, dropout_key = random.split(key, 2)
    variables = init({'params': key, "dropout": dropout_key}, jnp.ones(input_shape, jnp.float32))
    params, model_state = variables['params'], variables['batch_stats']
    return params, model_state


def compute_metrics(logits, labels):
    per_example_loss = cross_entropy(logits, labels, sparse=False)
    loss = per_example_loss.mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def train_step(apply_fn, state, batch, learning_rate_fn):
    def loss_fn(params):
        variables = {'params': params, **state.model_state}
        logits, new_model_state = apply_fn(
            variables, batch[0], training=True, mutable=['batch_stats'])
        loss = cross_entropy(logits, batch[1], sparse=False).mean()
        return loss, (new_model_state, logits)

    step = state.step
    optimizer = state.optimizer
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grad = grad_fn(optimizer.target)
    grad = lax.pmean(grad, axis_name='batch')
    new_model_state, logits = aux[1]
    new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
    metrics = compute_metrics(logits, batch[1])
    metrics['learning_rate'] = lr

    new_state = state.replace(
        step=step + 1, optimizer=new_optimizer, model_state=new_model_state)
    return new_state, metrics


def eval_step(apply_fn, state, batch):
    params = state.optimizer.target
    variables = {'params': params, **state.model_state}
    logits = apply_fn(
        variables, batch[0], training=False, mutable=False)
    return compute_metrics(logits, batch[1])


def prepare_tf_data(xs):
    def _prepare(x):
        x = x._numpy()
        return x.reshape((jax.local_device_count(), -1) + x.shape[1:])
    return jax.tree_map(_prepare, xs)


def sync_batch_stats(state):
    avg = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')

    new_model_state = state.model_state.copy({
        'batch_stats': avg(state.model_state['batch_stats'])})
    return state.replace(model_state=new_model_state)


def create_train_state(rng, model, shape):
    params, model_state = initialized(rng, shape, model)
    optimizer = optim.Momentum(beta=0.9, nesterov=True).create(params)
    state = TrainState(
        step=0, optimizer=optimizer, model_state=model_state)
    return state


@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)

    image, label = to_tensor(image, label)
    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

    label = tf.one_hot(label, 10)

    return image, label


rng = random.PRNGKey(0)

shape = (32, 32, 3)

batch_size = 128
eval_batch_size = batch_size * 16
ds_train, ds_test, steps_per_epoch, test_steps = make_cifar10_dataset(batch_size, eval_batch_size, transform)

model = ResNet(depth=10, k=2, dropout=0.3, num_classes=10)
state = create_train_state(rng, model, shape)
state = jax_utils.replicate(state)

base_lr = 0.1
epochs = 200
learning_rate_fn = cosine_lr(base_lr, steps_per_epoch, epochs, min_lr=0, warmup_epoch=0, warmup_min_lr=0.0)

p_train_step = jax.pmap(
    functools.partial(train_step, model.apply, learning_rate_fn=learning_rate_fn),
    axis_name='batch')
p_eval_step = jax.pmap(
    functools.partial(eval_step, model.apply), axis_name='batch')

train_iter = jax_utils.prefetch_to_device(map(prepare_tf_data, ds_train), 2)
eval_iter = jax_utils.prefetch_to_device(map(prepare_tf_data, ds_test), 2)

t_loop_start = time.time()
for epoch in range(epochs):
    epoch_metrics = []
    for step in range(steps_per_epoch):
        print(step)
        batch = next(train_iter)
        state, metrics = p_train_step(state, batch)
        epoch_metrics.append(metrics)
    epoch_metrics = common_utils.get_metrics(epoch_metrics)
    summary = jax.tree_map(lambda x: x.mean(), epoch_metrics)
    print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, summary['loss'], summary['accuracy'] * 100))

    eval_metrics = []
    state = sync_batch_stats(state)
    for _ in range(test_steps):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch)
        eval_metrics.append(metrics)
    eval_metrics = common_utils.get_metrics(eval_metrics)
    summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
    print('eval epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, summary['loss'], summary['accuracy'] * 100))
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
