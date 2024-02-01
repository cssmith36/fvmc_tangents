import os
import glob
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
from chex import ArrayTree
from ml_collections import ConfigDict

PyTree = Any

NuclConf = NuclCoord = Array
ElecCoord = ElecSpin = Array
ElecConf = Union[ElecCoord, Tuple[ElecCoord, ElecSpin]]
FullConf = Tuple[NuclConf, ElecConf]

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


@jax.custom_jvp
def log_cosh(x):
  """Numerically stable log_cosh, borrwoed from tfp."""
  # log(cosh(x)) = log(e^x + e^-x) - log(2).
  # For x > 0, we can rewrite this as x + log(1 + e^(-2 * x)) - log(2).
  # The second term will be small when x is large, so we don't get any large
  # cancellations.
  # Similarly for x < 0, we can rewrite the expression as -x + log(1 + e^(2 *
  # x)) - log(2)
  # This gives us abs(x) + softplus(-2 * abs(x)) - log(2)

  # For x close to zero, we can write the taylor series of softplus(
  # -2 * abs(x)) to see that we get;
  # log(2) - abs(x) + x**2 / 2. - x**4 / 12 + x**6 / 45. + O(x**8)
  # We can cancel out terms to get:
  # x ** 2 / 2.  * (1. - x ** 2 / 6) + x ** 6 / 45. + O(x**8)
  # For x < 45 * sixthroot(smallest normal), all higher level terms
  # disappear and we can use the above expression.
  from jax.nn import softplus
  abs_x = abs(x)
  logcosh = abs_x + softplus(-2 * abs_x) - jnp.log(2)
  bound = 45. * jnp.power(jnp.finfo(jnp.dtype(x)).tiny, 1 / 6.)
  return jnp.where(
      abs_x <= bound,
      jnp.exp(jnp.log(abs_x) + jnp.log1p(-jnp.square(abs_x) / 6.)),
      logcosh)

@log_cosh.defjvp
def _log_cosh_jvp(primals, tangents):
    # derivative of log(cosh(x)) is tanh(x)
    x, = primals
    x_dot, = tangents
    return log_cosh(x), x_dot * jnp.tanh(x)


def sample_genlogistic(key, a, b, shape=(), dtype=float):
    """Sample from type 4 generalized logistic distribution.

    See https://en.wikipedia.org/wiki/Generalized_logistic_distribution for details.
    Sample is generated as log(gamma_a) - log(gamma_b), where gamma_a and gamma_b
    are sampled from gamma distributions with parameters a and b, respectively.
    """
    keya, keyb = jax.random.split(key)
    loggamma_a = jax.random.loggamma(keya, a, shape=shape, dtype=dtype)
    loggamma_b = jax.random.loggamma(keyb, b, shape=shape, dtype=dtype)
    return loggamma_a - loggamma_b


def chol_qr(x, shift=None, psum_axis=None):
    *_, m, n = x.shape
    a = _H(x) @ x
    if psum_axis is not None:
        a = lax.psum(a, axis_name=psum_axis)
    if shift is None:
        shift = 1.2e-15 * (m*n + n*(n+1)) * a.trace(0,-1,-2).max()
    r = jsp.linalg.cholesky(a + shift * jnp.eye(n, dtype=x.dtype), lower=False)
    q = lax.linalg.triangular_solve(r, x, left_side=False, lower=False)
    return q, r


def fast_svd(a):
    """
    SVD using the eigen-decomposition of A A^T or A^T A,
    which appears to be much more efficient than jax.scipy.linalg.svd.
    """
    m, n = a.shape
    if m < n:
        a = _H(a)
    s2, v = jax.scipy.linalg.eigh(a.T.dot(a))
    s2, v = s2[::-1], v[:, ::-1]
    s = jnp.sqrt(jnp.abs(s2))
    u = a.dot(v/s)
    if m < n:
        return v, s, _H(u)
    return u, s, _H(v)


def r2c_grad(f, argnums=0, has_aux=False):
    if has_aux:
        return r2c_grad_with_aux(f, argnums=argnums)
    f_splited = compose(lambda x: jnp.array([x.real, x.imag]), f)
    def grad_f(*args, **kwargs):
        jac = jax.jacrev(f_splited, argnums=argnums)(*args, **kwargs)
        return jax.tree_map(lambda x: x[0] + 1j * x[1], jac)
    return grad_f

def r2c_grad_with_aux(f, argnums=0):
    f_splited = compose(lambda x: (jnp.array([x[0].real, x[0].imag]), x[1]), f)
    def grad_f(*args, **kwargs):
        jac, aux = jax.jacrev(f_splited,
                        argnums=argnums, has_aux=True)(*args, **kwargs)
        return jax.tree_map(lambda x: x[0] + 1j * x[1], jac), aux
    return grad_f

def adaptive_grad(f, argnums=0, has_aux=False):
    rgrad_f = jax.grad(f, argnums=argnums, has_aux=has_aux)
    cgrad_f = r2c_grad(f, argnums=argnums, has_aux=has_aux)
    def agrad_f(*args, **kwargs):
        try:
            return rgrad_f(*args, **kwargs)
        except TypeError:
            return cgrad_f(*args, **kwargs)
    return agrad_f


def wrap_complex_linear(func: Callable[[Array], Array]):
    def wrapped_func(x: Array):
        x_splited = jnp.stack([x.real, x.imag])
        f_splited = jax.vmap(func)(x_splited)
        return f_splited[0] + 1j * f_splited[1]
    return wrapped_func


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


def gen_kidx(n_d, n_k, close_shell=True):
    # n_d is spacial dimension
    # n_k is number of k points
    n_max = int(onp.ceil((n_k/2) ** (1/n_d)))
    grid = onp.arange(-n_max, n_max+1, dtype=int)
    mesh = onp.stack(onp.meshgrid(*([grid] * n_d), indexing='ij'), axis=-1)
    kall = mesh.reshape(-1, n_d)
    k2 = (kall ** 2).sum(-1)
    sidx = onp.lexsort((*kall.T[::-1], k2))
    if not close_shell:
        return kall[sidx[:n_k]]
    else:
        shell_select = k2[sidx] <= k2[sidx[n_k-1]]
        return kall[sidx[shell_select]]


def build_moving_avg(decay=0.99, early_growth=True):
    def moving_avg(acc, new, i=None):
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
    with jax.ensure_compile_time_eval():
        tmp = jax.tree_map(jnp.zeros, target_shape)
        flat, unravel_fn = ravel_pytree(tmp)
    return flat.size, unravel_fn


def tree_where(condition, x, y):
    return jax.tree_map(partial(jnp.where, condition), x, y)


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
    with jax.ensure_compile_time_eval():
        key = jax.random.PRNGKey(0)
        _trial_x = (jax.random.normal(key, (1024, 256)))
        y = actv_fn(_trial_x)
        gamma = y.var(axis=-1).mean() ** -0.5
    return gamma


def parse_activation(name, rescale=False, **kwargs):
    if callable(name):
        raw_actv_fn = name
    else:
        raw_actv_fn = getattr(nn, name)
    actv_fn = partial(raw_actv_fn, **kwargs)
    if rescale:
        gain = estimate_activation_gain(actv_fn) if rescale is True else rescale
        return lambda *x: actv_fn(*x) * gain
    else:
        return actv_fn


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


def backup_if_exist(filename, max_keep=None, prefix=""):
    idx_ptn = f"0{len(str(max_keep))}d" if max_keep is not None else "d"
    last_idx = max([0] + [int(ss.removeprefix(f"{filename}.{prefix}"))
                          for ss in glob.glob(f"{filename}.{prefix}*")]) + 1
    if max_keep is not None:
        last_idx = min(last_idx, max_keep - 1)
    fnames = [filename]
    fnames += [f"{filename}.{prefix}{ii:{idx_ptn}}"
               for ii in range(1, last_idx + 1)]
    for ii, fn in reversed(list(enumerate(fnames[:-1]))):
        if os.path.exists(fn):
            os.replace(fn, fnames[ii+1])


def save_checkpoint(filename, data, max_keep=1):
    backup_if_exist(filename, max_keep=max_keep)
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
    return jax.tree_map(cfg_to_dict, cfg.to_dict())

def cfg_to_yaml(cfg):
    import yaml
    from yaml import representer
    representer.Representer.add_representer(
        dict,
        lambda self, data: self.represent_mapping(
            'tag:yaml.org,2002:map', data, False))
    cdict = cfg_to_dict(cfg)
    def convert_obj(obj):
        if isinstance(obj, (Array, onp.ndarray)):
            return obj.tolist()
        from datetime import datetime
        yaml_type = (type(None), bool, int, float, complex, str, bytes,
                     list, tuple, set, dict, datetime,)
        if isinstance(obj, yaml_type):
            return obj
        return repr(obj)
    cdict = jax.tree_map(convert_obj, cdict)
    return yaml.dump(cdict, default_flow_style=None)

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


def parse_spin_num(n_el, spin):
    if spin is None:
        n_up, n_dn = n_el//2, n_el - n_el//2
    else:
        n_up = (n_el + spin) // 2
        n_dn = n_up - spin
    assert n_up + n_dn == n_el
    return n_up, n_dn


def split_spin(x: ElecConf) -> Tuple[ElecCoord, Optional[ElecSpin]]:
    if isinstance(x, tuple):
        return x
    else: # no spin info, return None for spin
        return x, None

def attach_spin(x: ElecCoord, s: Optional[ElecSpin]) -> ElecConf:
    if s is None:
        return x
    else:
        return x, s

def ensure_no_spin(x: ElecConf) -> ElecCoord:
    x, s = split_spin(x)
    if s is not None:
        raise ValueError("spin information is not allowed")
    return x


def collect_elems(elems):
    elems = onp.asarray(elems)
    assert elems.ndim == 1 and onp.all(onp.diff(elems) >= 0)
    uelems, counts = onp.unique(elems, return_counts=True)
    assert onp.all(onp.repeat(uelems, counts) == elems)
    return uelems, counts


def replicate_cell(pos, cell, copies):
    pos, cell, copies = list(map(onp.asarray, (pos, cell, copies)))
    assert pos.ndim == cell.ndim == 2
    assert pos.shape[-1] == cell.shape[0] == cell.shape[1] == len(copies)
    n_d = pos.shape[-1]
    XYZ = jnp.meshgrid(*[onp.arange(0, n_c) for n_c in copies], indexing="ij")
    xyz = jnp.stack(XYZ, axis=-1).reshape((-1, 1, n_d))
    disp = xyz @ cell
    return (pos + disp).reshape(-1, n_d), onp.diag(copies) @ cell


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
            the given axis, with shape (d, d') or (d,). If not provided, a simple sum
             is taken instead, equivalent to (d,) weights equal to 1. Defaults to None.
        axis (int, optional): axis along which to take the sum and max. Defaults to 0.
    Returns:
        Tuple[Array, Array]: sign of linear combination, log of linear
        combination. Both outputs have shape (..., d', ...), where d' = 1 if weights is
        None, and d' = weights.shape[1] otherwise.
    """
    max_val = jnp.max(vals, axis=axis, keepdims=True)
    shifted = signs * jnp.exp(vals - max_val)
    if weights is not None:
        w1d = weights.ndim == 1
        weights = weights[:, None] if w1d else weights
        # swap axis and -1 to conform to jnp.dot api
        shifted = jnp.swapaxes(shifted, axis, -1)
        result = shifted @ weights
        # swap axis and -1 back after the contraction
        result = jnp.swapaxes(result, axis, -1)
    else:
        w1d = True
        result = jnp.sum(shifted, axis=axis, keepdims=True)
    absres = jnp.abs(result)
    nsigns = result / absres if jnp.iscomplexobj(result) else jnp.sign(result)
    nvals = jnp.log(absres) + max_val
    if w1d:
        nsigns = jnp.squeeze(nsigns, axis=axis)
        nvals = jnp.squeeze(nvals, axis=axis)
    return nsigns, nvals


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
            scale = jnp.sqrt(2.) if self.rescale else 1.
            if self.residual and x.shape[-1] == tmp.shape[-1]:
                x = (x + tmp) / scale
            elif self.residual and x.shape[-1] * 2 == tmp.shape[-1]:
                x = (jnp.concatenate([x, x], axis=-1) + tmp) / scale
            else:
                x = tmp
        return x


def build_mlp(
    layer_sizes: Sequence[int],
    activation: Union[str, Callable] = "gelu",
    residual: bool = True,
    rescale: bool = False,
    last_bias: bool = False,
    **dense_kwargs
) -> Serial:
    layers = [nn.Dense(ls, **dense_kwargs) for ls in layer_sizes[:-1]]
    layers.append(nn.Dense(layer_sizes[-1],
                           **{**dense_kwargs, "use_bias": last_bias}))
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

    def p_func_if_pmap(obj, axis_name, **kwargs):
        try:
            jax.core.axis_frame(axis_name)
            return p_func(obj, axis_name, **kwargs)
        except NameError:
            return obj

    return p_func_if_pmap


pmax_if_pmap = wrap_if_pmap(lax.pmax)
pmin_if_pmap = wrap_if_pmap(lax.pmin)
psum_if_pmap = wrap_if_pmap(lax.psum)
pmean_if_pmap = wrap_if_pmap(lax.pmean)
all_gather_if_pmap = wrap_if_pmap(lax.all_gather)


@dataclasses.dataclass(frozen=True)
class PmapAxis:
    name : str
    def __post_init__(self):
        for nm, fn in (("vmap", jax.vmap), ("pmap", jax.pmap),
                       ("pmax", pmax_if_pmap), ("pmin", pmin_if_pmap),
                       ("psum", psum_if_pmap), ("pmean", pmean_if_pmap),
                       ("all_gather", all_gather_if_pmap)):
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
        object.__setattr__(self, "size", lambda: self.psum(1))

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
