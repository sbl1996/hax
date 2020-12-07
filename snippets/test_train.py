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
    optimizer: optim.Optimizer
    model_state: Any


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

@curry
def train_step(state, batch, model, learning_rate_fn, dropout_rng=None):
    inputs, targets = batch
    dropout_rng, new_dropout_rng = random.split(dropout_rng)

    def loss_fn(params):
        logits, new_model_state = model.apply(
            {'params': params, **state.model_state}, inputs, True,
            rngs={'dropout': dropout_rng}, mutable=['batch_stats'])
        loss = cross_entropy(logits, targets, sparse=False).mean()
        return loss, (new_model_state, logits)

    optimizer = state.optimizer
    step = optimizer.state.step
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grad = grad_fn(optimizer.target)
    grad = lax.pmean(grad, axis_name='batch')
    new_model_state, logits = aux[1]
    new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

    metrics = compute_metrics(logits, targets)
    metrics['learning_rate'] = lr

    new_state = state.replace(
        optimizer=new_optimizer, model_state=new_model_state)
    return new_state, metrics, new_dropout_rng


def eval_step(state, batch, model):
    params = state.optimizer.target
    inputs, targets = batch
    variables = {'params': params, **state.model_state}
    logits = model.apply(variables, inputs, False)
    return compute_metrics(logits, targets)


def prepare_tf_data(xs):
    def _prepare(x):
        x = x._numpy()
        return x.reshape((jax.local_device_count(), -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def sync_batch_stats(state):
    avg = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')

    new_model_state = state.model_state.copy({
        'batch_stats': avg(model_state['batch_stats'])})
    return state.replace(model_state=new_model_state)


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
rng, model_rng = random.split(rng)
rng, dropout_rng = random.split(rng)

shape = (32, 32, 3)

batch_size = 128
eval_batch_size = batch_size * 16
ds_train, ds_test, steps_per_epoch, test_steps = make_cifar10_dataset(batch_size, eval_batch_size, transform)

model = ResNet(depth=10, k=2, dropout=0.3, num_classes=10)

def initialize_variables(model, shape, init_rng):
    @jax.jit
    def init(*args):
        return model.init(*args)
    input_shape = (1,) + shape
    variables = init(init_rng, jnp.ones(input_shape, jnp.float32))
    params, model_state = variables['params'], variables['batch_stats']
    return params, model_state

init_rng = {"params": model_rng, "dropout": dropout_rng}
params, model_state = initialize_variables(model, shape, init_rng)
optimizer = optim.Momentum(beta=0.9, nesterov=True).create(params)

state = TrainState(optimizer=optimizer, model_state=model_state)
state = jax_utils.replicate(state)

base_lr = 0.1
epochs = 200
learning_rate_fn = cosine_lr(base_lr, steps_per_epoch, epochs, min_lr=0, warmup_epoch=0, warmup_min_lr=0.0)

p_train_step = jax.pmap(
    partial(train_step, model=model, learning_rate_fn=learning_rate_fn),
    axis_name='batch')
p_eval_step = jax.pmap(partial(eval_step, model=model), axis_name='batch')

train_iter = jax_utils.prefetch_to_device(map(prepare_tf_data, ds_train), 2)
eval_iter = jax_utils.prefetch_to_device(map(prepare_tf_data, ds_test), 2)

dropout_rngs = random.split(rng, jax.local_device_count())

t_loop_start = time.time()
for epoch in range(epochs):
    epoch_metrics = []
    for step in range(steps_per_epoch):
        print(step)
        batch = next(train_iter)
        state, metrics, dropout_rngs = p_train_step(state, batch, dropout_rng=dropout_rngs)
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
