import optax


def cosine_lr(
    base_lr, steps_per_epoch, epochs, min_lr=0, warmup_epoch=0, warmup_min_lr=0.0):
    warmup_fn = optax.linear_schedule(
        init_value=warmup_min_lr, end_value=base_lr,
        transition_steps=warmup_epoch * steps_per_epoch)
    cosine_epochs = max(epochs - warmup_epoch, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=cosine_epochs * steps_per_epoch,
        alpha=min_lr / base_lr)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epoch * steps_per_epoch])
    return schedule_fn