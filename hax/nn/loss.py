import jax
import jax.numpy as jnp


def cross_entropy(logits, labels, sparse=False, label_smoothing=0.0):
    num_classes = logits.shape[-1]
    if sparse:
        labels = jax.nn.one_hot(labels, num_classes, dtype=logits.dtype)
    if label_smoothing:
        labels = labels * (1 - label_smoothing) + label_smoothing / num_classes
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(labels * log_probs, axis=-1)
    return loss
