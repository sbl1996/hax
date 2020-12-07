import jax.numpy as jnp


def cosine_lr(base_lr, steps_per_epoch, epochs, min_lr=0, warmup_epoch=0, warmup_min_lr=0.0):
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epoch * steps_per_epoch

    def learning_rate_fn(step):

        factor = step / warmup_steps
        lr1 = warmup_min_lr + factor * (base_lr - warmup_min_lr)

        factor = jnp.cos((step - warmup_steps) * jnp.pi / (total_steps - warmup_steps)) * 0.5 + 0.5
        lr2 = min_lr + factor * (base_lr - min_lr)

        lr3 = min_lr
        return jnp.select(
            [jnp.less(step, warmup_steps),
             jnp.logical_or(jnp.greater_equal(step, warmup_steps), jnp.less(step, total_steps)),
             jnp.greater_equal(step, total_steps)],
            [lr1, lr2, lr3],
        )

    return learning_rate_fn
