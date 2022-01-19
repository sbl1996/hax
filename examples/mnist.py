from functools import partial
from toolz import curry

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
    iterator = map(prepare_tf_data, dataset)
    iterator = jax_utils.prefetch_to_device(iterator, 2)
    return iterator


def create_learning_rate_fn(
    base_lr, steps_per_epoch, epochs, warmup_epochs):
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_lr,
        transition_steps=warmup_epochs * steps_per_epoch)
    cosine_epochs = max(epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn


class CNN(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


@jax.jit
def apply_model(state, images, labels):

    def loss_fn(params):
        logits = CNN().apply({'params': params}, images)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


def cross_entropy_loss(logits, labels):
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    metrics = {
      'loss': loss,
      'acc': accuracy,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def train_step(state, batch, weight_decay):

    images, labels = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        loss = cross_entropy_loss(logits, labels)

        weight_penalty_params = jax.tree_leaves(params)
        weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1])
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name='batch')

    metrics = compute_metrics(logits, labels)
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


def eval_step(state, batch):
    images, labels = batch
    logits = state.apply_fn(
        {'params': state.params}, images)
    metrics = compute_metrics(logits, labels)
    return metrics


def train_epoch(p_train_step, state, train_it, steps_per_epoch):
    train_metrics = []
    for i in range(steps_per_epoch):
        state, metrics = p_train_step(state, next(train_it))
        train_metrics.append(metrics)
    train_metrics = common_utils.get_metrics(train_metrics)
    summary = {
        k: v for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
    }
    return state, summary


def eval_epoch(p_eval_step, state, test_iter, test_steps):
    eval_metrics = []
    for i in range(test_steps):
        metrics = p_eval_step(state, next(test_iter))
        eval_metrics.append(metrics)
    eval_metrics = common_utils.get_metrics(eval_metrics)
    summary = {
        k: v for k, v in jax.tree_map(lambda x: x.mean(), eval_metrics).items()
    }
    return summary


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

cnn = CNN()
params = cnn.init(init_rng, jnp.ones([1, 32, 32, 3]))['params']
base_lr = 0.1
epochs = 200
lr_schedule = create_learning_rate_fn(base_lr, steps_per_epoch, epochs, warmup_epochs=5)
tx = optax.sgd(lr_schedule, momentum=0.9, nesterov=True)

state = train_state.TrainState.create(
    apply_fn=cnn.apply, params=params, tx=tx)

state = jax_utils.replicate(state)

p_train_step = jax.pmap(partial(train_step, weight_decay=5e-4), axis_name='batch')
p_eval_step = jax.pmap(eval_step, axis_name='batch')

train_it = get_iter(ds_train)
test_it = get_iter(ds_test)
for epoch in range(epochs):
    print("Epoch %d/%d" % (epoch + 1, epochs))
    state, metrics = train_epoch(p_train_step, state, train_it, steps_per_epoch)
    log_metrics('train', metrics)

    metrics = eval_epoch(p_eval_step, state, test_it, test_steps)
    log_metrics('valid', metrics)