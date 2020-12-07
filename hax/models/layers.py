from difflib import get_close_matches
from functools import partial
from typing import Any, Sequence, Union, Tuple, Optional, List, Mapping
from cerberus import Validator

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers as init

from hax.models.bn import BatchNorm
from hax.models.dropout import Dropout

ModuleDef = Any

DEFAULTS = {
    'bn': {
        'momentum': 0.9,
        'eps': 1e-5,
        'affine': True,
        'track_running_stats': True,
        'sync': False,
    },
    'gn': {
        'groups': None,
        'channels_per_group': 16,
        'eps': 1e-5,
        'affine': True,
    },
    'activation': 'relu',
    'leaky_relu': {
        'alpha': 0.1,
    },
    'norm': 'bn',
    'init': {
        'type': 'msra',
        'mode': 'fan_in',
        'distribution': 'uniform',
    },
    'dtype': jnp.float32,
}

_defaults_schema = {
    'bn': {
        'momentum': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'eps': {'type': 'float', 'min': 0.0},
        'affine': {'type': 'boolean'},
        'track_running_stats': {'type': 'boolean'},
        'sync': {'type': 'boolean'},
    },
    'gn': {
        'eps': {'type': 'float', 'min': 0.0},
        'affine': {'type': 'boolean'},
        'groups': {'type': 'integer'},
        'channels_per_group': {'type': 'integer'},
    },
    'activation': {'type': 'string', 'allowed': ['relu', 'swish', 'mish', 'leaky_relu', 'sigmoid']},
    'leaky_relu': {
        'alpha': {'type': 'float', 'min': 0.0, 'max': 1.0},
    },
    'norm': {'type': 'string', 'allowed': ['bn', 'gn', 'none']},
    'init': {
        'type': {'type': 'string', 'allowed': ['msra', 'normal']},
        'mode': {'type': 'string', 'allowed': ['fan_in', 'fan_out']},
        'distribution': {'type': 'string', 'allowed': ['uniform', 'truncated_normal','untruncated_normal']},
    },
}


def set_defaults(kvs: Mapping):
    def _set_defaults(kvs, prefix):
        for k, v in kvs.items():
            if isinstance(v, dict):
                _set_defaults(v, prefix + (k,))
            else:
                set_default(prefix + (k,), v)

    return _set_defaults(kvs, ())


def set_default(keys: Union[str, Sequence[str]], value):
    def loop(d, keys, schema):
        k = keys[0]
        if k not in d:
            match = get_close_matches(k, d.keys())
            if match:
                raise KeyError("No such key `%s`, maybe you mean `%s`" % (k, match[0]))
            else:
                raise KeyError("No key `%s` in %s" % (k, d))
        if len(keys) == 1:
            v = Validator({k: schema[k]})
            if not v.validate({k: value}):
                raise ValueError(v.errors)
            d[k] = value
        else:
            loop(d[k], keys[1:], schema[k])

    if isinstance(keys, str):
        keys = [keys]
    loop(DEFAULTS, keys, _defaults_schema)


class Sequential(nn.Module):
    layers: List[nn.Module]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def calc_same_padding(kernel_size, dilation):
    kh, kw = kernel_size
    dh, dw = dilation
    ph = (kh + (kh - 1) * (dh - 1) - 1) // 2
    pw = (kw + (kw - 1) * (dw - 1) - 1) // 2
    padding = (ph, pw)
    return padding


def Conv2d(channels: int,
           kernel_size: Union[int, Tuple[int, int]],
           stride: Union[int, Tuple[int, int]] = 1,
           padding: Union[str, int, Tuple[int, int]] = 'same',
           groups: int = 1,
           dilation: int = 1,
           bias: Optional[bool] = None,
           norm: Optional[str] = None,
           act: Optional[str] = None):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(padding, str):
        assert padding == 'same'
    if padding == 'same':
        padding = calc_same_padding(kernel_size, dilation)
    if isinstance(padding, tuple) and len(padding) == 2:
        padding = [padding, padding]

    # Init
    init_cfg = DEFAULTS['init']
    if init_cfg['type'] == 'msra':
        mode = init_cfg['mode']
        distribution = init_cfg['distribution']
        kernel_init = init.variance_scaling(2.0, mode, distribution)
    else:
        raise ValueError("Unsupported init type: %s" % init_cfg['type'])

    bias_init = init.zeros

    if bias is None:
        use_bias = norm is None
    else:
        use_bias = bias

    conv = nn.Conv(channels, kernel_size=kernel_size, strides=stride, padding=padding,
                   kernel_dilation=dilation, use_bias=use_bias, feature_group_count=groups,
                   dtype=DEFAULTS['dtype'], kernel_init=kernel_init, bias_init=bias_init)

    layers = [conv]
    if norm:
        layers.append(Norm(norm))
    if act:
        layers.append(Act(act))

    if len(layers) == 1:
        return layers[0]
    else:
        return Sequential(layers)


def Act(type='default'):
    if type in ['default', 'def']:
        return Act(DEFAULTS['activation'])
    elif type == 'relu':
        return nn.relu
    elif type == 'swish':
        return nn.swish
    elif type == 'leaky_relu':
        return partial(nn.leaky_relu(negative_slope=DEFAULTS['leaky_relu']['alpha']))
    else:
        raise ValueError("Not supported activation: %s")


def get_groups(channels, ref=32):
    if channels == 1:
        return 1
    xs = filter(lambda x: channels % x == 0, range(2, channels + 1))
    c = min(xs, key=lambda x: abs(x - ref))
    if c < 8:
        c = max(c, channels // c)
    return channels // c


def Norm(type='default', **kwargs):
    affine = kwargs.get("affine")
    if type in ['default', 'def']:
        type = DEFAULTS['norm']
    if type == 'bn':
        cfg = DEFAULTS['bn']
        affine = affine or cfg['affine']
        track_running_stats = kwargs.get("track_running_stats")or cfg['track_running_stats']
        bn = BatchNorm(
            track_running_stats=track_running_stats, momentum=cfg['momentum'], epsilon=cfg['eps'],
            dtype=DEFAULTS['dtype'], use_bias=affine, use_scale=affine,

        )
        return bn
    elif type == 'gn':
        cfg = DEFAULTS['gn']
        affine = affine or cfg['affine']
        groups = kwargs.get("groups") or cfg['groups']
        gn = nn.GroupNorm(
            num_groups=groups, epsilon=cfg['eps'], dtype=DEFAULTS['dtype'],
            use_bias=affine, use_scale=affine,
        )
        return gn
    elif type == 'none':
        return Identity
    else:
        raise ValueError("Unsupported normalization type: %s" % type)


def Pool2d(kernel_size, stride, padding='same', type='avg'):
    assert padding == 0 or padding == 'same'
    if padding == 0:
        padding = 'valid'
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    if type == 'avg':
        pool = nn.avg_pool
    elif type == 'max':
        pool = nn.max_pool
    else:
        raise ValueError("Unsupported pool type: %s" % type)

    return partial(pool, window_shape=kernel_size, strides=stride, padding=padding)


def GlobalAvgPool(x, keep_dim=False):
    return jnp.mean(x, axis=(1, 2), keepdims=keep_dim)


def Identity(x):
    return x


def Linear(channels, act=None):
    kernel_init = init.kaiming_uniform()
    dense = nn.Dense(channels, kernel_init=kernel_init, bias_init=init.zeros)
    if act:
        return Sequential([
            dense,
            Act(act),
        ])
    else:
        return dense


def mish(x):
    return x * nn.tanh(nn.softplus(x))
