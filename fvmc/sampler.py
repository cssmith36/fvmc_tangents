import dataclasses
from functools import partial
from typing import Callable, Dict, NamedTuple, Tuple, Union, Optional

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from optax import bias_correction, update_moment

from .utils import (PMAP_AXIS_NAME, Array, ArrayTree, PmapAxis, PyTree,
                    adaptive_split, build_moving_avg, clip_gradient,
                    parse_spin_num, ravel_shape, tree_where)
from .utils import log_cosh, sample_genlogistic # for relavistic hmc

KeyArray = Array
Params = ArrayTree
Sample = ArrayTree
State = Tuple[Sample, ...] # first element of state is always current sample
Data = Tuple[Sample, Array] # data is a tuple of sample and its log prob
Info = Dict[str, PyTree] # info is a dict containing stats of the sampling process
Flag = Union[int, str]


class MCSampler(NamedTuple):
    # to jit the sampler, use jax.tree_map(jax.jit, sampler)
    sample: Callable[[KeyArray, Params, State], Tuple[State, Data, Info]]
    init: Callable[[KeyArray, Params], State]
    refresh: Callable[[State, Params], State]

    def __call__(self, key: KeyArray, params: Params, state: State):
        """Call the sample function. See `self.sample` for details."""
        return self.sample(key, params, state)

    def burn_in(self, key: KeyArray, params: Params, state: State, steps: int):
        """Burn in the state for given steps"""
        # inner = lambda s,k: (self.sample(k, params, s)[0], None)
        # return lax.scan(inner, state, jax.random.split(key, steps))[0]
        for _ in range(steps):
            key, subkey = adaptive_split(key, multi_device=key.ndim>1)
            state = self.sample(subkey, params, state)[0]
        return state


def choose_sampler_builder(name: str) -> Callable[..., MCSampler]:
    name = name.lower()
    if name in ("gaussian",):
        return build_gaussian
    if name in ("metropolis", "mcmc", "mh"):
        return build_metropolis
    if name in ("langevin", "mala"):
        return build_langevin
    if name in ("hamiltonian", "hybrid", "hmc"):
        return build_hamiltonian
    if name in ("black", "blackjax"):
        return build_blackjax
    raise NotImplementedError(f"unsupported sampler type: {name}")


def choose_adaptive_builder(name: str,
                            harmonic=False, **kwargs) -> Callable[..., MCSampler]:
    name = name.lower()
    if name in ("metropolis", "mcmc", "mh"):
        keyword = "sigma"
        default_target = 0.5
    elif name in ("langevin", "mala"):
        keyword = "tau"
        default_target = 0.65
    elif name in ("hamiltonian", "hybrid", "hmc"):
        keyword = "dt"
        default_target = 0.9
    else:
        raise NotImplementedError(f"unsupported adaptive sampler type: {name}")
    target = kwargs.pop("target", default_target)
    if harmonic:
        reference = "recip_ratio"
        increasing = False
        transform = lambda x: 1./x
    else:
        reference = "is_accepted"
        increasing = False
        transform = None
    new_kwargs = dict(keyword=keyword, reference=reference, target=target,
                      increasing=increasing, transform=transform, **kwargs)
    raw_builder = choose_sampler_builder(name)
    return make_adaptive(raw_builder, **new_kwargs)


def build_sampler(log_prob_fn: Callable[[Params, Sample], Array],
                  shape_or_init: Union[tuple, np.ndarray, callable],
                  name: str,
                  adaptive: Union[None, bool, dict] = None,
                  beta: float = 1,
                  **kwargs):
    if adaptive is not None and adaptive is not False:
        adaptive = {} if adaptive is True else adaptive
        builder = choose_adaptive_builder(name, **adaptive)
    else:
        builder = choose_sampler_builder(name)
    logdens_fn = lambda p, x: beta * log_prob_fn(p, x)
    return builder(logdens_fn, shape_or_init, **kwargs)


def build_conf_init_fn(elems, nuclei, n_elec,
                       sigma_x=1., with_r=False, sigma_r=0.1):
    if nuclei.size == 0:
        nuclei = nuclei.sum(-2, keepdims=True)
        elems = [0]
    elems = jnp.asarray(elems, dtype=int)
    n_dim = nuclei.shape[-1]
    n_elec = int(sum(n_elec)) if not isinstance(n_elec, int) else n_elec
    if elems.sum() != n_elec: # put extra charge in first atoms
        elems = elems.at[0].add(n_elec - elems.sum())
    elec_list = [parse_spin_num(el, el%2 * (-1)**ii) for ii, el in enumerate(elems)]

    def init_fn(key):
        xa, xb = [], []
        for (na, nb), nuc in zip(elec_list, nuclei):
            key, ska, skb = jax.random.split(key, 3)
            xa.append(nuc + jax.random.normal(ska, (na, n_dim)) * sigma_x)
            xb.append(nuc + jax.random.normal(skb, (nb, n_dim)) * sigma_x)
        init_x = jnp.concatenate(xa + xb, axis=0)[:n_elec]
        if not with_r:
            return init_x
        else:
            init_r = nuclei + jax.random.normal(key, nuclei.shape) * sigma_r
            return [init_r, init_x]

    return init_fn


##### Below are sampler transformations #####

MC_BATCH_AXIS_NAME = "_mc_batch_axis"

def make_batched(sampler: MCSampler, n_batch: int, concat: bool = False,
                 vmap_axis_name: str = MC_BATCH_AXIS_NAME):
    sample_fn, init_fn, refresh_fn = sampler
    vaxis = PmapAxis(vmap_axis_name)
    def sample(key, params, state):
        vkey = jax.random.split(key, n_batch)
        new_state, *res = vaxis.vmap(sample_fn, (0, None, 0))(vkey, params, state)
        if concat:
            res = jax.tree_map(jnp.concatenate, res)
        return new_state, *res
    def init(key, params):
        vkey = jax.random.split(key, n_batch)
        return vaxis.vmap(init_fn, (0, None))(vkey, params)
    refresh = vaxis.vmap(refresh_fn, (0, None))
    return MCSampler(sample, init, refresh)


def make_multistep(sampler: MCSampler, n_step: int, concat: bool = False):
    sample_fn, init_fn, refresh_fn = sampler
    multisample_fn = make_multistep_fn(sample_fn, n_step, concat)
    return MCSampler(multisample_fn, init_fn, refresh_fn)


def make_multistep_fn(sample_fn, n_step, concat=False):
    def _split_output(out): # to satisfy scan requirement
        return out[0], out[1:]
    def multi_sample(key, params, state):
        inner = lambda s,k: _split_output(sample_fn(k, params, s))
        keys = jax.random.split(key, n_step)
        new_state, res = lax.scan(inner, state, keys)
        if concat:
            res = jax.tree_map(jnp.concatenate, res)
        return new_state, *res
    return multi_sample


def make_chained(*samplers):
    # chain multiple sampler, always refresh
    def sample(key, params, state):
        info = {}
        sample, aux_list = state
        for ii, splr in enumerate(samplers):
            inner = (sample, *aux_list[ii])
            inner = splr.refresh(inner, params)
            inner, data, infoi = splr.sample(key, params, inner)
            sample, *aux = inner
            aux_list[ii] = aux
            info[f"part_{ii}"] = infoi
        state = sample, aux_list
        return state, data, info
    # save all aux informations in state
    def init(key, params):
        aux_list = []
        for splr in samplers:
            sample, *aux = splr.init(key, params)
            aux_list.append(aux)
        return sample, aux_list
    # dummy refresh, actual one is in sample
    def refresh(state, params):
        return state
    return MCSampler(sample, init, refresh)


def make_adaptive(
        sampler_factory: Callable[..., MCSampler],
        keyword: str, # the kwarg name to be tuned upon
        reference: str, # the info key to be used as reference
        target: Union[float, Tuple[float, float]], # the targeting interval for the reference
        transform: Optional[Callable[[float], float]] = None, # transform the reference
        increasing: bool = False, # if the reference is increasing w.r.t. the kwarg
        *, # above are required, below are optional user settings
        interval: int = 100, # the interval to update the kwarg
        scale_factor: float = 1.01, # the factor to scale the kwarg
        ema_decay: Optional[float] = None, # the decay of the exponential moving average
        vmap_axis_name: str = MC_BATCH_AXIS_NAME,
        pmap_axis_name: str = PMAP_AXIS_NAME,
) -> Callable[..., MCSampler]:

    # initialize pmap and vmap utilities
    vaxis = PmapAxis(vmap_axis_name)
    paxis = PmapAxis(pmap_axis_name)

    # exponential moving average of the reference
    _decay = 1 - 1. / interval if ema_decay is None else ema_decay
    moving_avg = build_moving_avg(_decay)

    # dynamic tuning function
    assert scale_factor > 1., "scale factor must be larger than 1"
    if not increasing:
        scale_factor = 1. / scale_factor
    scale_arr = jnp.array([scale_factor, 1., 1./scale_factor])
    tmin, tmax = (target, target) if isinstance(target, float) else target
    def tuning(value, reference):
        # 0: too small, 1: good, 2: too large
        idx = (reference > tmin).astype(int) + (reference > tmax).astype(int)
        return value * scale_arr[idx]

    # transformation of the reference
    transform = transform or (lambda x: x)

    # return a new sampler factory that takes the same arguments as the original
    def adaptive_builder(logdens_fn, shape_or_init, **kwargs):
        raw_sampler = sampler_factory(logdens_fn, shape_or_init, **kwargs)
        # build new sampler every time
        def sample(key, params, state):
            inner_sample, inner_aux, count, adapt_value, ema_ref = state
            inner_state = (inner_sample, *inner_aux)
            ada_kwargs = {**kwargs, keyword: adapt_value}
            ada_sampler = sampler_factory(logdens_fn, shape_or_init, **ada_kwargs)
            new_inner_state, data, info = ada_sampler.sample(key, params, inner_state)
            ref_update = transform(paxis.pmean(vaxis.all_mean(info[reference])))
            new_ema_ref = moving_avg(ema_ref, ref_update, count)
            adapt_value = jnp.where((count > 0) & (count % interval == 0),
                                    tuning(adapt_value, new_ema_ref),
                                    adapt_value)
            new_sample, *new_aux = new_inner_state
            new_state = (new_sample, new_aux, count+1, adapt_value, new_ema_ref)
            info["tuned_hparam"] = adapt_value
            return new_state, data, info
        # refresh the inner state
        def refresh(state, params):
            inner_sample, inner_aux, count, adapt_value, ema_ref = state
            new_sample, *new_aux = raw_sampler.refresh((inner_sample, *inner_aux), params)
            return new_sample, new_aux, count, adapt_value, ema_ref
        # init inner state and wrap it with adapting kwarg
        def init(key, params):
            count = 0
            inner_sample, *inner_aux = raw_sampler.init(key, params)
            init_value = kwargs.get(keyword, 0.1) # defaults to 0.1
            ema_ref = 0.
            return (inner_sample, inner_aux, count, init_value, ema_ref)
        # assemble the adaptive sampler
        return MCSampler(sample, init, refresh)

    return adaptive_builder


##### Below are generation functions for different samplers #####

def build_gaussian(logdens_fn, shape_or_init, mu=0., sigma=1., truncate=None):
    sample_shape = _extract_sample_shape(shape_or_init)
    xsize, unravel = ravel_shape(sample_shape)
    info = {"is_accepted": True, "recip_ratio": 1.}

    def sample(key, params, state):
        if truncate is not None:
            trc = jnp.abs(truncate)
            rawgs = jax.random.truncated_normal(key, -trc, trc, (xsize,))
        else:
            rawgs = jax.random.normal(key, (xsize,))
        new_sample = rawgs * sigma + mu
        new_logdens = logd_gaussian(new_sample, mu, sigma).sum()
        return (new_sample,), (unravel(new_sample), new_logdens), info

    def init(key, params):
        return (jnp.zeros((xsize,)),)

    def refresh(state, params):
        return (state[0],)

    return MCSampler(sample, init, refresh)


def build_metropolis(logdens_fn, shape_or_init, sigma=0.1, steps=10, mass=1.):
    sample_shape = _extract_sample_shape(shape_or_init)
    xsize, unravel = ravel_shape(sample_shape)
    ravel_logd = lambda p, x: logdens_fn(p, unravel(x))
    mass = _align_vector(mass, sample_shape)
    sigma = sigma / jnp.sqrt(mass)

    def step(key, params, state):
        x1, ld1 = state
        gkey, ukey = jax.random.split(key)
        x2 = x1 + sigma * jax.random.normal(gkey, shape=x1.shape)
        ld2 = ravel_logd(params, x2)
        ratio = ld2 - ld1
        return (*mh_select(ukey, ratio, state, (x2, ld2)), ratio)

    def sample(key, params, state):
        multi_step = make_multistep_fn(step, steps, concat=False)
        new_state, accepted, log_ratio = multi_step(key, params, state)
        new_sample, new_logdens = new_state
        info = {"is_accepted": accepted.mean(),
                "recip_ratio": _recip_ratio(log_ratio).mean()}
        return new_state, (unravel(new_sample), new_logdens), info

    def refresh(state, params):
        sample = state[0]
        ld_new = ravel_logd(params, sample)
        return (sample, ld_new)

    init = _gen_init_from_refresh(refresh, shape_or_init)

    return MCSampler(sample, init, refresh)


def build_langevin(logdens_fn, shape_or_init, tau=0.1, steps=10,
                   mass=1., grad_clipping=None):
    # shape and constants
    sample_shape = _extract_sample_shape(shape_or_init)
    xsize, unravel = ravel_shape(sample_shape)
    mass = _align_vector(mass, sample_shape)
    tau = tau / mass
    # prepare functions
    if grad_clipping is not None: grad_clipping *= jnp.sqrt(mass)
    ravel_logd = lambda p, x: logdens_fn(p, unravel(_gclip(x, grad_clipping)))
    logd_and_grad = jax.value_and_grad(ravel_logd, 1)

    # log transition probability q(x2|x1)
    def log_q(x2, x1, g1):
        d = x2 - x1 - tau * g1
        norm2 = (d * d.conj()).real
        return (-1/(4*tau) * norm2).sum(-1)

    def step(key, params, state):
        x1, g1, ld1 = state
        gkey, ukey = jax.random.split(key)
        x2 = x1 + tau*g1 + jnp.sqrt(2*tau)*jax.random.normal(gkey, shape=x1.shape)
        ld2, g2 = logd_and_grad(params, x2)
        g2 = g2.conj() # handle complex grads, no influence for real case
        ratio = ld2 + log_q(x1, x2, g2) - ld1 - log_q(x2, x1, g1)
        return (*mh_select(ukey, ratio, state, (x2, g2, ld2)), ratio)

    def sample(key, params, state):
        multi_step = make_multistep_fn(step, steps, concat=False)
        new_state, accepted, log_ratio = multi_step(key, params, state)
        new_sample, new_grads, new_logdens = new_state
        info = {"is_accepted": accepted.mean(),
                "recip_ratio": _recip_ratio(log_ratio).mean()}
        return new_state, (unravel(new_sample), new_logdens), info

    def refresh(state, params):
        sample = state[0]
        ld_new, grads_new = logd_and_grad(params, sample)
        return (sample, grads_new.conj(), ld_new)

    init = _gen_init_from_refresh(refresh, shape_or_init)

    return MCSampler(sample, init, refresh)


def build_hamiltonian(logdens_fn, shape_or_init, dt=0.1, length=1.,
                      steps=None, segments=1, mass=1., speed_limit=None,
                      jitter_dt=False, grad_clipping=None, div_threshold=1000.):
    # shape information
    sample_shape = _extract_sample_shape(shape_or_init)
    xsize, unravel = ravel_shape(sample_shape)
    # align mass and rescale
    mass = _align_vector(mass, sample_shape)
    scale = jnp.sqrt(mass)
    # prepare functions
    if grad_clipping is not None: grad_clipping *= scale
    ravel_logd = lambda p, x: logdens_fn(p, unravel(_gclip(x, grad_clipping)))
    logd_and_grad = jax.value_and_grad(ravel_logd, 1)
    # kinetic energy function
    if speed_limit is None:
        ke_fn = lambda p: -logd_gaussian(p, 0, 1).sum(-1)
        draw_momentum = jax.random.normal
    else:
        ke_fn = lambda p: relavistic_ke(p, speed_limit).sum(-1)
        draw_momentum = partial(sample_relavistic_momentum, c=speed_limit)
    # align length
    if steps is not None:
        if length is not None:
            from . import LOGGER
            LOGGER.warning("Both steps and length are given in hmc, use steps.")
        length = steps * dt
    length /= segments
    # function to get jittered dt
    jitter_dt = float(jitter_dt)
    _jdt = lambda k: ((dt * (1 + (jax.random.uniform(k) - 0.5) * 2 * jitter_dt))
                      if 0 < jitter_dt <= 1 else dt)

    def step(key, params, state):
        gkey, ukey, tkey = jax.random.split(key, 3)
        q1, g1, ld1, *extras = state
        # scaled coordinates to account for mass
        q1s, f1s, v1 = q1 * scale, -g1 / scale, -ld1
        p1 = draw_momentum(gkey, shape=(xsize,))
        # pe fn take scaled q as input
        pe_fn = lambda xs: -ravel_logd(params, xs / scale)
        # determine integration length and prepare leap frog
        leapfrog = gen_leapfrog(pe_fn, _jdt(tkey), round(length / dt), kinetic_fn=ke_fn,
                                with_carry=True, threshold=div_threshold)
        # the actual integration
        q2s, p2, f2s, v2 = leapfrog(q1s, p1, f1s, v1)
        # scale back the coordinates for output
        q2, g2, ld2 = q2s / scale, -f2s * scale, -v2
        # metroplis-hastings selection according to the hamiltonian
        log_ratio = (-ke_fn(-p2) - v2) - (-ke_fn(p1) - v1)
        (qn, gn, ldn), accepted = mh_select(ukey, log_ratio, state[:3], (q2, g2, ld2))
        return (qn, gn, ldn, *extras), accepted, log_ratio

    def sample(key, params, state):
        multi_step = make_multistep_fn(step, segments, concat=False)
        new_state, accepted, log_ratio = multi_step(key, params, state)
        new_sample, new_grads, new_logdens, *extras = new_state
        info = {"is_accepted": accepted.mean(),
                "recip_ratio": _recip_ratio(log_ratio).mean()}
        return new_state, (unravel(new_sample), new_logdens), info

    def refresh(state, params):
        sample = state[0]
        extras = state[3:]
        ld_new, grads_new = logd_and_grad(params, sample)
        return (sample, grads_new.conj(), ld_new, *extras)

    init = _gen_init_from_refresh(refresh, shape_or_init)

    return MCSampler(sample, init, refresh)


def build_blackjax(logdens_fn, shape_or_init, kernel="nuts", grad_clipping=None, **kwargs):
    from blackjax import hmc, nuts
    sample_shape = _extract_sample_shape(shape_or_init)
    xsize, unravel = ravel_shape(sample_shape)
    inv_mass = jnp.asarray(kwargs.pop("inverse_mass_matrix", 0.5 * jnp.ones(xsize)))
    ravel_logd = lambda p, x: logdens_fn(p, unravel(_gclip(x, grad_clipping)))
    kmodule = {"hmc": hmc, "nuts": nuts}[kernel]

    def sample(key, params, state):
        logprob_fn = partial(ravel_logd, params)
        kernel = kmodule(logprob_fn,
            inverse_mass_matrix=inv_mass, **kwargs)
        state = state[1]
        state, info = kernel.step(key, state)
        info_dict = {**info._asdict(), "recip_ratio": 1./info.acceptance_rate}
        return ((state.position, state),
                (unravel(state.position), state.logdensity), info_dict)

    def refresh(state, params):
        sample = state[0]
        logprob_fn = partial(ravel_logd, params)
        return (sample, kmodule.init(sample, logprob_fn))

    init = _gen_init_from_refresh(refresh, shape_or_init)

    return MCSampler(sample, init, refresh)


##### Below are helper functions for samplers #####

def logd_gaussian(x, mu=0., sigma=1.):
    """unnormalized log density of Gaussian distribution"""
    return -0.5 * jnp.square((x - mu) / sigma)


def mh_select(key, ratio, state1, state2):
    rnd = jnp.log(jax.random.uniform(key, shape=ratio.shape))
    cond = ratio > rnd
    new_state = tree_where(cond, state2, state1)
    return new_state, cond


def gen_leapfrog(potential_fn, dt, steps, kinetic_fn=None,
                 with_carry=True, threshold=1000.):
    kinetic_fn = kinetic_fn or (lambda p: -logd_gaussian(p, 0, 1).sum(-1))
    velocity_fn = jax.grad(kinetic_fn)
    pe_and_grad = jax.value_and_grad(potential_fn)

    def leapfrog_carry(q, p, g, v):
        # p for momentom and q for position
        # g for grad (neg force) and v for potential
        # simple Euler integration step
        e0 = v + kinetic_fn(p)
        def integral_step(carry):
            i, q, p, g, v, _div = carry
            p -= 0.5 * dt * g # half p step
            q += dt * velocity_fn(p) # whole q step
            v, g = pe_and_grad(q)
            g = g.conj()
            p -= 0.5 * dt * g # half p step
            k = kinetic_fn(p)
            _div = jnp.abs(k + v - e0) > threshold
            return i+1, q, p, g, v, _div
        def loop_cond(carry):
            i, _, _, _, _, _div = carry
            return (i < steps) & ~_div
        # leapfrog by shifting half step
        carry = (0, q, p, g, v, False)
        _, q, p, g, v, _ = lax.while_loop(loop_cond, integral_step, carry)
        return q, p, g, v

    def leapfrog_nocarry(q, p):
        v, g = pe_and_grad(q)
        return leapfrog_carry(q, p, g, v)[:2]

    return leapfrog_carry if with_carry else leapfrog_nocarry


def relavistic_ke(p, c):
    r"""relavistic kinetic energy, use formula $K(p) = c^2 \log(\cosh(p / c))$

    It is relativistic in the sense that the velocity $dK/dp = c \tanh(p / c)$
    is bounded between -c and c, and is roughly linear for small $p$.
    """
    return c**2 * log_cosh(p / c)


def sample_relavistic_momentum(key, c, shape=(), dtype=float):
    r"""Sample momentum from relativistic distribution

    The distribution is defined by the kinetic energy $K(p) = c^2 \log(\cosh(p / c))$.
    Hence the momentum is sampled from the distribution $p \sim \exp(-K(p))$, which
    is a generalized logistic distribution with scale $c/2$, and $a = b = c^2/2$.
    """
    return sample_genlogistic(key, c**2/2, c**2/2, shape, dtype) * c / 2


def _recip_ratio(log_ratio):
    return jnp.exp(-jnp.minimum(0, log_ratio))


def _gclip(x, bnd):
    if bnd is None:
        return x
    if isinstance(bnd, (int, float, Array)):
        bnd = (-abs(bnd), abs(bnd))
    return clip_gradient(x, *bnd)


def _extract_sample_shape(shape_or_init):
    # shape_or_init is either a pytree of shapes
    # or a init function that take a key and give an init x
    with jax.ensure_compile_time_eval():
        if not callable(shape_or_init): # is shape
            sample_shape = shape_or_init
        else: # is init function
            _dummy_key = jax.random.PRNGKey(0)
            init_sample = shape_or_init(_dummy_key)
            sample_shape = jax.tree_map(lambda a: np.array(a.shape), init_sample)
    return sample_shape


def _gen_init_from_refresh(refresh_fn, shape_or_init):
    # shape_or_init is either a pytree of shapes
    # or a init function that take a key and give an init x
    with jax.ensure_compile_time_eval():
        if not callable(shape_or_init):
            size, unravel = ravel_shape(shape_or_init)
            mu, sigma = 0., 1.
            raw_init = lambda key: jax.random.normal(key, (size,)) * sigma + mu
        else:
            from jax.flatten_util import ravel_pytree
            raw_init = lambda key: ravel_pytree(shape_or_init(key))[0]

    def init(key, params):
        sample = raw_init(key)
        return refresh_fn((sample,), params)

    return init


def _align_vector(vec, sample_shape):
    xsize, unravel = ravel_shape(sample_shape)
    vec_ = ravel_pytree(vec)[0]
    vec = (vec_ if vec_.size in (1, xsize) else
           ravel_pytree(jax.tree_map(jnp.broadcast_to, vec, sample_shape))[0])
    assert vec.shape in ((1,), (xsize,))
    return vec
