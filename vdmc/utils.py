import jax
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax.tree_util import tree_map
from flax import linen as nn
from ml_collections import ConfigDict
from typing import Dict, Sequence, Union, Callable, Any, Optional
from functools import partial, reduce
import dataclasses
import pickle
import time
import os

from jax.numpy import ndarray as Array
PyTree = Any

_t_real = float
_t_cplx = complex

# _t_real = jnp.float32
# _t_cplx = jnp.complex64

# _t_real = jnp.float64
# _t_cplx = jnp.comple128


def adaptive_split(key, num=2, multi_device=False):
    if multi_device:
        import kfac_jax
        return kfac_jax.utils.p_split_num(key, num)
    else:
        return jax.random.split(key, num)


def compose(*funcs):
    def c2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))
    return reduce(c2, funcs)


def ith_output(func, i):
    def warpped(*args, **kwargs):
        return func(*args, **kwargs)[i]
    return warpped


def just_grad(x):
    return x - lax.stop_gradient(x)


@jax.custom_vjp
def clip_gradient(x, g_min=None, g_max=None):
  return x  # identity function

def clip_gradient_fwd(x, lo=None, hi=None):
  return x, (lo, hi)  # save bounds as residuals

def clip_gradient_bwd(res, g):
  lo, hi = res
  return (jnp.clip(g, lo, hi), None, None)  # use None to indicate zero cotangents for lo and hi

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)


def _T(x): 
    return jnp.swapaxes(x, -1, -2)

def _H(x): 
    return jnp.conj(_T(x))


def symmetrize(x): 
    return (x + _H(x)) / 2


def cmult(x1, x2):
    return ((x1.real * x2.real - x1.imag * x2.imag) 
        + 1j * (x1.imag * x2.real + x1.real * x2.imag))


def diffmat(xa, xb):
    return jnp.expand_dims(xa, -2) - jnp.expand_dims(xb, -3)

def pdist(x):
    # x is assumed to have dimension [..., n, 3]
    n = x.shape[-2]
    diff = diffmat(x, x)
    diff_padded = diff + jnp.eye(n)[..., None]
    dist = jnp.linalg.norm(diff_padded, axis=-1) * (1 - jnp.eye(n))
    return dist

def cdist(xa, xb):
    diff = diffmat(xa, xb)
    dist = jnp.linalg.norm(diff, axis=-1)
    return dist


def build_moving_avg(decay=0.99, early_growth=True):
    def moving_avg(acc, new, i):
        if early_growth:
            iteration_decay = jnp.minimum(decay, (1.0 + i) / (10.0 + i))
        else:
            iteration_decay = decay
        updated_acc = iteration_decay * acc
        updated_acc += (1 - iteration_decay) * new
        return jax.lax.stop_gradient(updated_acc)
    return moving_avg


def ravel_shape(target_shape):
    from jax.flatten_util import ravel_pytree
    tmp = tree_map(jnp.zeros, target_shape)
    flat, unravel_fn = ravel_pytree(tmp)
    return flat.size, unravel_fn


def tree_where(condition, x, y):
    return tree_map(partial(jnp.where, condition), x, y)


def fix_init(key, value, dtype=None, random=0., rnd_additive=False):
    value = jnp.asarray(value, dtype=dtype)
    if random <= 0.:
        return value
    else:
        perturb = jax.random.truncated_normal(
            key, -2, 2, value.shape, _t_real) * random
        if rnd_additive:
            return value + perturb
        else:
            return value * (1 + perturb)


def parse_activation(name, **kwargs):
    if not isinstance(name, str):
        return name
    raw_fn = getattr(nn, name)
    return partial(raw_fn, **kwargs)


def parse_bool(keys, inputs):
    if isinstance(keys, str):
        return parse_bool((keys,), inputs)[keys]
    res_dict = {}
    if isinstance(inputs, str) and inputs.lower() in ("all", "true"):
        inputs = True
    if isinstance(inputs, str) and inputs.lower() in ("none", "false"):
        inputs = False
    if isinstance(inputs, bool):
        for key in keys:
            res_dict[key] = inputs
    else:
        for key in keys:
            res_dict[key] = key in inputs
    return res_dict


def ensure_mapping(obj, default_key="name"):
    try:
        return dict(**obj)
    except TypeError:
        return {default_key: obj}


def save_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_checkpoint(filename, data, keep=1):
    numlen = len(str(keep))
    fnames = [filename]
    fnames += [f"{filename}.{ii:0{numlen}d}" for ii in range(1, keep)]
    for ii, fn in reversed(list(enumerate(fnames[:-1]))):
        if os.path.exists(fn):
            os.replace(fn, fnames[ii+1])
    save_pickle(filename, data)


def cfg_to_dict(cfg):
    if not isinstance(cfg, ConfigDict):
        return cfg
    return tree_map(cfg_to_dict, cfg.to_dict())

def cfg_to_yaml(cfg):
    import yaml
    from yaml import representer
    representer.Representer.add_representer(
        dict,
        lambda self, data: self.represent_mapping(
            'tag:yaml.org,2002:map', data, False))
    return yaml.dump(cfg_to_dict(cfg), default_flow_style=None)

def dict_to_cfg(cdict, **kwargs):
    if not isinstance(cdict, (dict, ConfigDict)):
        return cdict
    tree_type = (tuple, list)
    cfg = ConfigDict(cdict, **kwargs)
    for k, v in cfg.items():
        if isinstance(v, ConfigDict):
            cfg[k] = dict_to_cfg(v, **kwargs)
        if type(v) in tree_type:
            cfg[k] = type(v)(dict_to_cfg(vi, **kwargs) for vi in v)
    return cfg
    

class Serial(nn.Module):
    layers : Sequence[nn.Module]
    residual : bool = True
    activation : Union[str, Callable] = "gelu"

    @nn.compact
    def __call__(self, x):
        actv = parse_activation(self.activation)
        for i, lyr in enumerate(self.layers):
            tmp = lyr(x)
            if i != len(self.layers) - 1:
                tmp = actv(tmp)
            if self.residual:
                if x.shape[-1] >= tmp.shape[-1]:
                    x = x[...,:tmp.shape[-1]] + tmp
                else:
                    x = tmp.at[...,:x.shape[-1]].add(x)
            else:
                x = tmp
        return x


def build_mlp(
    layer_sizes : Sequence[int],
    residual : bool = True,
    activation : Union[str, Callable] = "gelu",
    **dense_kwargs
) -> Serial:
    layers = [nn.Dense(ls, **dense_kwargs) for ls in layer_sizes]
    return Serial(layers, residual=residual, activation=activation)


class Printer:

    def __init__(self, 
                 field_format: Dict[str, Optional[str]], 
                 time_format: Optional[str]=None,
                 **print_kwargs):
        all_format = {**field_format, "time": time_format}
        all_format = {k: v for k, v in all_format.items() if v is not None}
        self.fields = all_format
        self.header = "\t".join(self.fields.keys())
        self.format = "\t".join(f"{{{k}:{v}}}" for k, v in self.fields.items())
        self.kwargs = print_kwargs
        self._tick = time.perf_counter()

    def print_header(self, prefix: str = ""):
        print(prefix+self.header, **self.kwargs)

    def print_fields(self, field_dict: Dict[str, Any], prefix: str = ""):
        output = self.format.format(**field_dict, time=time.perf_counter()-self._tick)
        print(prefix+output, **self.kwargs)

    def reset_timer(self):
        self._tick = time.perf_counter()


def wrap_if_pmap(p_func):

    def p_func_if_pmap(obj, axis_name):
        try:
            jax.core.axis_frame(axis_name)
            return p_func(obj, axis_name)
        except NameError:
            return obj

    return p_func_if_pmap


pmax_if_pmap = wrap_if_pmap(lax.pmax)
pmin_if_pmap = wrap_if_pmap(lax.pmin)
psum_if_pmap = wrap_if_pmap(lax.psum)
pmean_if_pmap = wrap_if_pmap(lax.pmean)


@dataclasses.dataclass(frozen=True)
class PmapAxis:
    name : str
    def __post_init__(self):
        for nm, fn in (("vmap", jax.vmap), ("pmap", jax.pmap),
                       ("pmax", pmax_if_pmap), ("pmin", pmin_if_pmap),
                       ("psum", psum_if_pmap), ("pmean", pmean_if_pmap)):
            object.__setattr__(self, nm, partial(fn, axis_name=self.name))
        for nm in ("max", "min", "sum", "mean"):
            jnp_fn = getattr(jnp, nm)
            pax_fn = getattr(self, f"p{nm}")
            all_fn = compose(pax_fn, jnp_fn)
            object.__setattr__(self, f"all_{nm}", all_fn)

PMAP_AXIS_NAME = "_pmap_axis"
PAXIS = PmapAxis(PMAP_AXIS_NAME)
