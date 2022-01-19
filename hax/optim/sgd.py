from typing import Optional, Union, Any, Callable

from optax import GradientTransformation, chain, trace, identity, add_decayed_weights, Params
from optax._src.alias import _scale_by_learning_rate


def sgd(
    learning_rate,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[Params], Any]]] = None,
    accumulator_dtype: Optional[Any] = None,
) -> GradientTransformation:
  """A canonical Stochastic Gradient Descent optimiser.

  This implements stochastic gradient descent. It also includes support for
  momentum, and nesterov acceleration, as these are standard practice when
  using stochastic gradient descent to train deep neural networks.

  References:
    Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

  Args:
    learning_rate: this is a fixed global scaling factor.
    momentum: (default `None`), the `decay` rate used by the momentum term,
      when it is set to `None`, then momentum is not used at all.
    nesterov (default `False`): whether nesterov momentum is used.
    weight_decay: strength of the weight decay regularization.
    mask: a tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.
    accumulator_dtype: optional `dtype` to be used for the accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation`.
  """
  return chain(
      (trace(decay=momentum, nesterov=nesterov,
                       accumulator_dtype=accumulator_dtype)
       if momentum is not None else identity()),
      add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate)
  )