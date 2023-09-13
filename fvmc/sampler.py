import dataclasses
from functools import partial
from typing import Callable, Dict, NamedTuple, Tuple, Union

import jax
import numpy as onp
from jax import lax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from .utils import (PMAP_AXIS_NAME, Array, ArrayTree, PmapAxis, PyTree,
                    adaptive_split, clip_gradient, parse_spin, ravel_shape,
                    tree_map, tree_where)
from .wavefunction import nn

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


def build_sampler(log_prob_fn: Callable[[Params, Sample], Array],
                  shape_or_init: Union[tuple, onp.ndarray, callable],
                  name: str,
                  beta: float = 1,
                  **kwargs):
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
    elec_list = [parse_spin(el, el%2 * (-1)**ii) for ii, el in enumerate(elems)]

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
            res = tree_map(jnp.concatenate, res)
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
            res = tree_map(jnp.concatenate, res)
        return new_state, *res
    return multi_sample


def make_chained(*samplers):
    init = samplers[-1].init
    def sample(key, params, state):
        info = {}
        for ii, splr in enumerate(samplers):
            state = splr.refresh(state, params)
            state, data, infoi = splr.sample(key, params, state)
            info[f"part_{ii}"] = infoi
        return state, data, info
    def refresh(state, params):
        return state
    return MCSampler(sample, init, refresh)


##### Below are generation functions for different samplers #####

def build_gaussian(logdens_fn, shape_or_init, mu=0., sigma=1., truncate=None):
    sample_shape = _extract_sample_shape(shape_or_init)
    xsize, unravel = ravel_shape(sample_shape)
    info = {"is_accepted": True}

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


def build_metropolis(logdens_fn, shape_or_init, sigma=0.1, steps=10):
    sample_shape = _extract_sample_shape(shape_or_init)
    xsize, unravel = ravel_shape(sample_shape)
    ravel_logd = lambda p, x: logdens_fn(p, unravel(x))

    def step(key, params, state):
        x1, ld1 = state
        gkey, ukey = jax.random.split(key)
        x2 = x1 + sigma * jax.random.normal(gkey, shape=x1.shape)
        ld2 = ravel_logd(params, x2)
        ratio = ld2 - ld1
        return mh_select(ukey, ratio, state, (x2, ld2))

    def sample(key, params, state):
        multi_step = make_multistep_fn(step, steps, concat=False)
        new_state, accepted = multi_step(key, params, state)
        new_sample, new_logdens = new_state
        info = {"is_accepted": accepted.mean()}
        return new_state, (unravel(new_sample), new_logdens), info

    def refresh(state, params):
        sample = state[0]
        ld_new = ravel_logd(params, sample)
        return (sample, ld_new)

    init = _gen_init_from_refresh(refresh, shape_or_init)

    return MCSampler(sample, init, refresh)


def build_langevin(logdens_fn, shape_or_init, tau=0.1, steps=10, grad_clipping=None):
    sample_shape = _extract_sample_shape(shape_or_init)
    xsize, unravel = ravel_shape(sample_shape)
    ravel_logd = lambda p, x: logdens_fn(p, unravel(_gclip(x, grad_clipping)))
    logd_and_grad = jax.value_and_grad(ravel_logd, 1)

    # log transition probability q(x2|x1)
    def log_q(x2, x1, g1):
        d = x2 - x1 - tau * g1
        norm = (d * d.conj()).real.sum(-1)
        return -1/(4*tau) * norm

    def step(key, params, state):
        x1, g1, ld1 = state
        gkey, ukey = jax.random.split(key)
        x2 = x1 + tau*g1 + jnp.sqrt(2*tau)*jax.random.normal(gkey, shape=x1.shape)
        ld2, g2 = logd_and_grad(params, x2)
        g2 = g2.conj() # handle complex grads, no influence for real case
        ratio = ld2 + log_q(x1, x2, g2) - ld1 - log_q(x2, x1, g1)
        return mh_select(ukey, ratio, state, (x2, g2, ld2))

    def sample(key, params, state):
        multi_step = make_multistep_fn(step, steps, concat=False)
        new_state, accepted = multi_step(key, params, state)
        new_sample, new_grads, new_logdens = new_state
        info = {"is_accepted": accepted.mean()}
        return new_state, (unravel(new_sample), new_logdens), info

    def refresh(state, params):
        sample = state[0]
        ld_new, grads_new = logd_and_grad(params, sample)
        return (sample, grads_new.conj(), ld_new)

    init = _gen_init_from_refresh(refresh, shape_or_init)

    return MCSampler(sample, init, refresh)


def build_hamiltonian(logdens_fn, shape_or_init, dt=0.1, length=1.,
                      mass=1., jittered=False, grad_clipping=None,
                      vmap_axis_name=MC_BATCH_AXIS_NAME,
                      pmap_axis_name=PMAP_AXIS_NAME):
    # initialize pmap and vmap utilities
    vaxis = PmapAxis(vmap_axis_name)
    paxis = PmapAxis(pmap_axis_name)
    # prepare functions and constants
    sample_shape = _extract_sample_shape(shape_or_init)
    xsize, unravel = ravel_shape(sample_shape)
    ravel_logd = lambda p, x: logdens_fn(p, unravel(_gclip(x, grad_clipping)))
    logd_and_grad = jax.value_and_grad(ravel_logd, 1)
    mass = _align_vector(mass, sample_shape)
    scale = jnp.sqrt(mass)

    def _jitter_length(extra_state):
        if not jittered:
            return length, extra_state
        jkey, subkey = jax.random.split(extra_state[0])
        jlength = (1 - float(jittered) * jax.random.uniform(subkey)) * length
        return jlength, (jkey, *extra_state[1:])

    def sample(key, params, state):
        gkey, ukey = jax.random.split(key)
        q1, g1, ld1, *extras = state
        # scaled coordinates to account for mass
        q1s, g1s = q1 * scale, g1 / scale
        p1 = jax.random.normal(gkey, shape=q1.shape)
        # pot fn take scaled q as input
        potential_fn = lambda xs: -ravel_logd(params, xs / scale)
        # determine integration length
        length, extras = _jitter_length(extras)
        leapfrog = gen_leapfrog(potential_fn, dt, round(length / dt), True)
        q2s, p2, f2s, v2 = leapfrog(q1s, p1, -g1s, -ld1)
        # scale back the coordinates for output
        q2, g2, ld2 = q2s / scale, -f2s * scale, -v2
        ratio = (logd_gaussian(-p2).sum(-1)+ld2) - (logd_gaussian(p1).sum(-1)+ld1)
        (qn, gn, ldn), accepted = mh_select(ukey, ratio, state[:3], (q2, g2, ld2))
        info = {"is_accepted": accepted}
        return (qn, gn, ldn, *extras), (unravel(qn), ldn), info

    def refresh(state, params):
        sample = state[0]
        extras = state[3:]
        ld_new, grads_new = logd_and_grad(params, sample)
        return (sample, grads_new.conj(), ld_new, *extras)

    raw_init = _gen_init_from_refresh(refresh, shape_or_init)
    def init(key, params):
        key, jkey = jax.random.split(key)
        raw_state = raw_init(key, params)
        extras = []
        if jittered:
            extras.append(paxis.pmin(vaxis.pmin(jkey)))
        state = (*raw_state, *extras)
        return state

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
        return ((state.position, state),
               (unravel(state.position), -state.potential_energy), info._asdict())

    def refresh(state, params):
        sample = state[0]
        logprob_fn = partial(ravel_logd, params)
        return (sample, kmodule.init(sample, logprob_fn))

    init = _gen_init_from_refresh(refresh, shape_or_init)

    return MCSampler(sample, init, refresh)


##### Below are helper functions for samplers #####

def logd_gaussian(x, mu=0., sigma=1.):
    """unnormalized log density of Gaussian distribution"""
    return -0.5 * ((x - mu) / sigma) ** 2


def mh_select(key, ratio, state1, state2):
    rnd = jnp.log(jax.random.uniform(key, shape=ratio.shape))
    cond = ratio > rnd
    new_state = tree_where(cond, state2, state1)
    return new_state, cond


def gen_leapfrog(potential_fn, dt, steps, with_carry=True):
    pot_and_grad = jax.value_and_grad(potential_fn)

    def leapfrog_carry(q, p, g, v):
        # p for momentom and q for position
        # g for grad (neg force) and v for potential
        # simple Euler integration step
        def int_step(i, carry):
            q, p = carry
            q += dt * p
            p -= dt * pot_and_grad(q)[1].conj()
            return (q, p)
        # leapfrog by shifting half step
        p -= 0.5 * dt * g # first half p
        (q, p) = lax.fori_loop(1, steps, int_step, (q, p))
        q += dt * p # final whole step update of q
        v, g = pot_and_grad(q)
        g = g.conj() # for potential complex variables
        p -= 0.5 * dt * g # final half p
        return q, p, g, v

    def leapfrog_nocarry(q, p):
        v, g = pot_and_grad(q)
        return leapfrog_carry(q, p, g, v)[:2]

    return leapfrog_carry if with_carry else leapfrog_nocarry


def _gclip(x, bnd):
    if bnd is None:
        return x
    if isinstance(bnd, (int, float, Array)):
        bnd = (-abs(bnd), abs(bnd))
    return clip_gradient(x, *bnd)


def _extract_sample_shape(shape_or_init):
    # shape_or_init is either a pytree of shapes
    # or a init function that take a key and give an init x
    if not callable(shape_or_init): # is shape
        sample_shape = shape_or_init
    else: # is init function
        _dummy_key = jax.random.PRNGKey(0)
        init_sample = shape_or_init(_dummy_key)
        sample_shape = jax.tree_map(lambda a: onp.array(a.shape), init_sample)
    return sample_shape


def _gen_init_from_refresh(refresh_fn, shape_or_init):
    # shape_or_init is either a pytree of shapes
    # or a init function that take a key and give an init x
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
