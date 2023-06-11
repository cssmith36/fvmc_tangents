import os
import pickle
import time
import dataclasses
from functools import partial, reduce
from typing import Any, Callable, Dict, Optional, Sequence, Union, Tuple

import jax
import numpy as onp
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from flax import linen as nn
from jax.numpy import ndarray as Array
from jax.tree_util import tree_map
from chex import ArrayTree
from ml_collections import ConfigDict

PyTree = Any

_t_real = float
_t_cplx = complex

# _t_real = jnp.float32
# _t_cplx = jnp.complex64

# _t_real = jnp.float64
# _t_cplx = jnp.complex128


PMAP_AXIS_NAME = "_pmap_axis"


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


def displace_matrix(xa, xb, disp_fn=None):
    if disp_fn is None:
        return jnp.expand_dims(xa, -2) - jnp.expand_dims(xb, -3)
    else:
        return jax.vmap(jax.vmap(disp_fn, (None, 0)), (0, None))(xa, xb)

def pdist(x, disp_fn=None):
    # x is assumed to have dimension [..., n, 3]
    n = x.shape[-2]
    disp = displace_matrix(x, x, disp_fn)
    disp_padded = disp + jnp.eye(n)[..., None]
    dist = jnp.linalg.norm(disp_padded, axis=-1) * (1 - jnp.eye(n))
    return dist

def cdist(xa, xb, disp_fn=None):
    disp = displace_matrix(xa, xb, disp_fn)
    dist = jnp.linalg.norm(disp, axis=-1)
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


def estimate_activation_gain(actv_fn):
    key = jax.random.PRNGKey(0)
    _trial_x = (jax.random.normal(key, (1024, 256)))
    y = actv_fn(_trial_x)
    gamma = y.var(axis=-1).mean() ** -0.5
    return gamma


_activation_gain_dict = {(actv_fn := getattr(nn, name)): 
                            estimate_activation_gain(actv_fn)
                         for name in ("silu", "tanh", "gelu")}

def parse_activation(name, rescale=False, **kwargs):
    if callable(name):
        actv_fn = name
    else:
        actv_fn = getattr(nn, name)
    if rescale:
        gain = _activation_gain_dict[actv_fn] if rescale is True else rescale
        return lambda *x: actv_fn(*x, **kwargs) * gain
    else:
        return partial(actv_fn, **kwargs)


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


def multi_process_name(filename):
    n_proc = jax.process_count()
    if n_proc == 1:
        return filename
    numlen = len(str(n_proc))
    return f"{filename}.pid{jax.process_index():0{numlen}d}" 
    

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


def parse_spin(n_el, spin):
    if spin is None:
        n_up, n_dn = n_el//2, n_el - n_el//2
    else:
        n_up = (n_el + spin) // 2
        n_dn = n_up - spin
    assert n_up + n_dn == n_el
    return n_up, n_dn


def collect_elems(elems):
    elems = onp.asarray(elems)
    assert elems.ndim == 1 and onp.all(onp.diff(elems) >= 0)
    uelems, counts = onp.unique(elems, return_counts=True)
    assert onp.all(onp.repeat(uelems, counts) == elems)
    return uelems, counts


def adaptive_residual(x, y, rescale=False):
    if y.shape != x.shape:
        return y
    scale = jnp.sqrt(2.) if rescale else 1.
    return (x + y) / scale


def exp_shifted(x, normalize=None, pmap_axis_name=PMAP_AXIS_NAME):
    paxis = PmapAxis(pmap_axis_name)
    stblz = paxis.all_max(lax.stop_gradient(x))
    exp = jnp.exp(x - stblz)
    if normalize:
        assert normalize.lower() in ("sum", "mean"), "invalid normalize option"
        reducer = getattr(paxis, f"all_{normalize.lower()}")
        total = reducer(lax.stop_gradient(exp))
        exp /= total
        stblz += jnp.log(total)
    return exp, stblz


def log_linear_exp(
    signs: Array,
    vals: Array,
    weights: Optional[Array] = None,
    axis: int = 0,
) -> Tuple[Array, Array]:
    """Stably compute sign and log(abs(.)) of sum_i(sign_i * w_ij * exp(vals_i)) + b_j.
    In order to avoid overflow when computing
        log(abs(sum_i(sign_i * w_ij * exp(vals_i)))),
    the largest exp(val_i) is divided out from all the values and added back in after
    the outer log, i.e.
        log(abs(sum_i(sign_i * w_ij * exp(vals_i - max)))) + max.
    This trick also avoids the underflow issue of when all vals are small enough that
    exp(val_i) is approximately 0 for all i.
    Args:
        signs (Array): array of signs of the input x with shape (..., d, ...),
            where d is the size of the given axis
        vals (Array): array of log|abs(x)| with shape (..., d, ...), where d is
            the size of the given axis
        weights (Array, optional): weights of a linear transformation to apply to
            the given axis, with shape (d, d'). If not provided, a simple sum is taken
            instead, equivalent to (d, 1) weights equal to 1. Defaults to None.
        axis (int, optional): axis along which to take the sum and max. Defaults to 0.
    Returns:
        Tuple[Array, Array]: sign of linear combination, log of linear
        combination. Both outputs have shape (..., d', ...), where d' = 1 if weights is
        None, and d' = weights.shape[1] otherwise.
    """
    max_val = jnp.max(vals, axis=axis, keepdims=True)
    terms_divided_by_max = signs * jnp.exp(vals - max_val)
    if weights is not None:
        # swap axis and -1 to conform to jnp.dot api
        terms_divided_by_max = jnp.swapaxes(terms_divided_by_max, axis, -1)
        transformed_divided_by_max = jnp.dot(terms_divided_by_max, weights)
        # swap axis and -1 back after the contraction
        transformed_divided_by_max = jnp.swapaxes(transformed_divided_by_max, axis, -1)
    else:
        transformed_divided_by_max = jnp.sum(
            terms_divided_by_max, axis=axis, keepdims=True
        )

    signs = jnp.sign(transformed_divided_by_max)
    vals = jnp.log(jnp.abs(transformed_divided_by_max)) + max_val
    return signs, vals


class Serial(nn.Module):
    layers: Sequence[nn.Module]
    residual: bool = True
    activation: Union[str, Callable] = "gelu"
    rescale: bool = False

    @nn.compact
    def __call__(self, x):
        actv = parse_activation(self.activation, rescale=self.rescale)
        for i, lyr in enumerate(self.layers):
            tmp = lyr(x)
            if i != len(self.layers) - 1:
                tmp = actv(tmp)
            if self.residual:
                if x.shape[-1] >= tmp.shape[-1]:
                    x = x[...,:tmp.shape[-1]] + tmp
                elif x.shape[-1] * 2 == tmp.shape[-1]:
                    x = jnp.concatenate([x, x], axis=-1) + tmp
                else:
                    x = tmp.at[...,:x.shape[-1]].add(x)
                if self.rescale:
                    x /= jnp.sqrt(2.)
            else:
                x = tmp
        return x


def build_mlp(
    layer_sizes: Sequence[int],
    residual: bool = True,
    activation: Union[str, Callable] = "gelu",
    rescale: bool = False,
    **dense_kwargs
) -> Serial:
    layers = [nn.Dense(ls, **dense_kwargs) for ls in layer_sizes]
    return Serial(layers, 
                  residual=residual, activation=activation, rescale=rescale)


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
            nan_fn = getattr(jnp, f'nan{nm}')
            allnan_fn = compose(pax_fn, nan_fn)
            object.__setattr__(self, f"all_nan{nm}", allnan_fn)
        object.__setattr__(self, "all_average", 
            lambda a, w: self.all_mean(a * w) / self.all_mean(w))
        object.__setattr__(self, "all_nanaverage", 
            lambda a, w: self.all_nansum(a * w)
                         / self.all_nansum(~jnp.isnan(a) * w))

PAXIS = PmapAxis(PMAP_AXIS_NAME)


PROTON_MASS = 1836.152673

# copied from PySCF
ISOTOPE_MAIN = onp.array([
    0  ,   # GHOST
    1  ,   # H
    4  ,   # He
    7  ,   # Li
    9  ,   # Be
    11 ,   # B
    12 ,   # C
    14 ,   # N
    16 ,   # O
    19 ,   # F
    20 ,   # Ne
    23 ,   # Na
    24 ,   # Mg
    27 ,   # Al
    28 ,   # Si
    31 ,   # P
    32 ,   # S
    35 ,   # Cl
    40 ,   # Ar
    39 ,   # K
    40 ,   # Ca
    45 ,   # Sc
    48 ,   # Ti
    51 ,   # V
    52 ,   # Cr
    55 ,   # Mn
    56 ,   # Fe
    59 ,   # Co
    58 ,   # Ni
    63 ,   # Cu
    64 ,   # Zn
    69 ,   # Ga
    74 ,   # Ge
    75 ,   # As
    80 ,   # Se
    79 ,   # Br
    84 ,   # Kr
    85 ,   # Rb
    88 ,   # Sr
    89 ,   # Y
    90 ,   # Zr
    93 ,   # Nb
    98 ,   # Mo
    98 ,   # 98Tc
    102,   # Ru
    103,   # Rh
    106,   # Pd
    107,   # Ag
    114,   # Cd
    115,   # In
    120,   # Sn
    121,   # Sb
    130,   # Te
    127,   # I
    132,   # Xe
    133,   # Cs
    138,   # Ba
    139,   # La
    140,   # Ce
    141,   # Pr
    144,   # Nd
    145,   # Pm
    152,   # Sm
    153,   # Eu
    158,   # Gd
    159,   # Tb
    162,   # Dy
    162,   # Ho
    168,   # Er
    169,   # Tm
    174,   # Yb
    175,   # Lu
    180,   # Hf
    181,   # Ta
    184,   # W
    187,   # Re
    192,   # Os
    193,   # Ir
    195,   # Pt
    197,   # Au
    202,   # Hg
    205,   # Tl
    208,   # Pb
    209,   # Bi
    209,   # Po
    210,   # At
    222,   # Rn
    223,   # Fr
    226,   # Ra
    227,   # Ac
    232,   # Th
    231,   # Pa
    238,   # U
    237,   # Np
    244,   # Pu
    243,   # Am
    247,   # Cm
    247,   # Bk
    251,   # Cf
    252,   # Es
    257,   # Fm
    258,   # Md
    259,   # No
    262,   # Lr
    261,   # Rf
    262,   # Db
    263,   # Sg
    262,   # Bh
    265,   # Hs
    266,   # Mt
    0  ,   # Ds
    0  ,   # Rg
    0  ,   # Cn
    0  ,   # Nh
    0  ,   # Fl
    0  ,   # Mc
    0  ,   # Lv
    0  ,   # Ts
    0  ,   # Og
])