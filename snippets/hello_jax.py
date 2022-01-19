from toolz import curry

import tensorflow as tf
import jax
from jax import random, numpy as jnp

from hanser.datasets.cifar import make_cifar100_dataset
from hanser.transform import random_crop, normalize, to_tensor

from hax.models.cifar.preactresnet import ResNet
from hax.optim.lr_schedule import cosine_lr

@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)

    image, label = to_tensor(image, label)
    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

    label = tf.one_hot(label, 10)

    return image, label


batch_size = 128
eval_batch_size = batch_size * 16
ds_train, ds_test, steps_per_epoch, test_steps = make_cifar100_dataset(
    batch_size, eval_batch_size, transform)

rng = random.PRNGKey(0)

image_size = 224


model = ResNet(depth=28, k=10, dropout=0.3, num_classes=100)

state = create_train_state(rng, config, model, image_size)
state = restore_checkpoint(state, workdir)
# step_offset > 0 if restarting from checkpoint
step_offset = int(state.step)
state = jax_utils.replicate(state)

base_lr = 0.1
epochs = 200
learning_rate_fn = cosine_lr(base_lr, steps_per_epoch, epochs)

p_train_step = jax.pmap(
    functools.partial(train_step, model.apply,
                      learning_rate_fn=learning_rate_fn),
    axis_name='batch')
p_eval_step = jax.pmap(
    functools.partial(eval_step, model.apply), axis_name='batch')

epoch_metrics = []
t_loop_start = time.time()
for step, batch in zip(range(step_offset, num_steps), train_iter):
    state, metrics = p_train_step(state, batch)
    epoch_metrics.append(metrics)
    if (step + 1) % steps_per_epoch == 0:
        epoch = step // steps_per_epoch
        epoch_metrics = common_utils.get_metrics(epoch_metrics)
        summary = jax.tree_map(lambda x: x.mean(), epoch_metrics)
        logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f',
                     epoch, summary['loss'], summary['accuracy'] * 100)
        steps_per_sec = steps_per_epoch / (time.time() - t_loop_start)
        t_loop_start = time.time()

        epoch_metrics = []
        eval_metrics = []

        # sync batch statistics across replicas
        state = sync_batch_stats(state)
        for _ in range(steps_per_eval):
            eval_batch = next(eval_iter)
            metrics = p_eval_step(state, eval_batch)
            eval_metrics.append(metrics)
        eval_metrics = common_utils.get_metrics(eval_metrics)
        summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
        logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                     epoch, summary['loss'], summary['accuracy'] * 100)
    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
        state = sync_batch_stats(state)

# Wait until computations are done before exiting
jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
