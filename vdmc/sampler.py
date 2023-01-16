import jax
from jax import lax
from jax import numpy as jnp
import dataclasses
import numpy as onp
from functools import partial
from typing import NamedTuple, Callable, Tuple, Union, Dict

from .wavefunction import nn
from .utils import PyTree, Array
from .utils import ravel_shape, tree_where, tree_map, clip_gradient


KeyArray = Array
Params = PyTree
Sample = PyTree
State = Tuple[Sample, ...] # first element of state is always current sample
Data = Tuple[Sample, Array] # data is a tuple of sample and its log prob
Flag = Union[int, str]


class MCSampler(NamedTuple):
    # to jit the sampler, use jax.tree_map(jax.jit, sampler)
    sample: Callable[[KeyArray, Params, State], Tuple[State, Data]]
    init: Callable[[KeyArray, Params], State]
    refresh: Callable[[State, Params], State]

    def __call__(self, key: KeyArray, params: Params, state: State):
        """Call the sample function. See `self.sample` for details."""
        return self.sample(key, params, state)

    def burn_in(self, key: KeyArray, params: Params, state: State, steps: int):
        """Burn in the state for given steps"""
        inner = lambda s,k: (self.sample(k, params, s)[0], None)
        return lax.scan(inner, state, jax.random.split(key, steps))[0]
    

def choose_sampler_maker(name: str) -> Callable[..., MCSampler]:
    name = name.lower()
    if name in ("gaussian",):
        return make_gaussian
    if name in ("metropolis", "mcmc", "mh"):
        return make_metropolis
    if name in ("langevin", "mala"):
        return make_langevin
    if name in ("hamiltonian", "hybrid", "hmc"):
        return make_hamiltonian
    if name in ("black", "blackjax"):
        return make_blackjax
    raise NotImplementedError(f"unsupported sampler type: {name}")


def make_sampler(model: nn.Module, n_elec: int, 
                 name: str, beta: float = 1, 
                 **kwargs):
    maker = choose_sampler_maker(name)
    logdens_fn = lambda p, x: 2 * beta * model.apply(p, x)[1]
    sample_shape = onp.array([n_elec, 3])
    return maker(logdens_fn, sample_shape, **kwargs)


##### Below are sampler transformations #####

def make_batched(sampler: MCSampler, n_batch: int, concat: bool = False):
    sample_fn, init_fn, refresh_fn = sampler
    def sample(key, params, state):
        vkey = jax.random.split(key, n_batch)
        new_state, data = jax.vmap(sample_fn, (0, None, 0))(vkey, params, state)
        if concat:
            data = tree_map(jnp.concatenate, data)
        return new_state, data
    def init(key, params):
        vkey = jax.random.split(key, n_batch)
        return jax.vmap(init_fn, (0, None))(vkey, params)
    refresh = jax.vmap(refresh_fn, (0, None))
    return MCSampler(sample, init, refresh)


def make_multistep(sampler: MCSampler, n_step: int, concat: bool = False):
    sample_fn, init_fn, refresh_fn = sampler
    multisample_fn = make_multistep_fn(sample_fn, n_step, concat)
    return MCSampler(multisample_fn, init_fn, refresh_fn)


def make_multistep_fn(sample_fn, n_step, concat=False):
    def multi_sample(key, params, state):
        inner = lambda s,k: sample_fn(k, params, s)
        keys = jax.random.split(key, n_step)
        new_state, data = lax.scan(inner, state, keys)
        if concat:
            data = tree_map(jnp.concatenate, data)
        return new_state, data
    return multi_sample


def make_chained(*samplers):
    init = samplers[-1].init
    def sample(key, params, state):
        for splr in samplers:
            state = splr.refresh(state, params)
            state, data = splr.sample(key, params, state)
        return state, data
    def refresh(state, params):
        return state
    return MCSampler(sample, init, refresh)


##### Below are generation functions for different samplers #####

def make_gaussian(logdens_fn, sample_shape, mu=0., sigma=1., truncate=None):
    xsize, unravel = ravel_shape(sample_shape)

    def sample(key, params, state):
        if truncate is not None:
            trc = jnp.abs(truncate)
            rawgs = jax.random.truncated_normal(key, -trc, trc, (xsize,))
        else:
            rawgs = jax.random.normal(key, (xsize,))
        new_sample = rawgs * sigma + mu
        new_logdens = logd_gaussian(new_sample, mu, sigma).sum()
        return (new_sample,), (unravel(new_sample), new_logdens)
    
    def init(key, params):
        return (jnp.zeros((xsize,)),)

    def refresh(state, params):
        return (state[0],)

    return MCSampler(sample, init, refresh)


def make_metropolis(logdens_fn, sample_shape, sigma=0.05, steps=10):
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
        return new_state, (unravel(new_sample), new_logdens)

    def refresh(state, params):
        sample = state[0]
        ld_new = ravel_logd(params, sample)
        return (sample, ld_new)
    
    init = _make_init_from_refresh(refresh, xsize)

    return MCSampler(sample, init, refresh)


def make_langevin(logdens_fn, sample_shape, tau=0.01, steps=10, grad_clipping=None):
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
        return new_state, (unravel(new_sample), new_logdens)

    def refresh(state, params):
        sample = state[0]
        ld_new, grads_new = logd_and_grad(params, sample)
        return (sample, grads_new.conj(), ld_new)

    init = _make_init_from_refresh(refresh, xsize)

    return MCSampler(sample, init, refresh)


def make_hamiltonian(logdens_fn, sample_shape, dt=0.1, length=1., grad_clipping=None):
    xsize, unravel = ravel_shape(sample_shape)
    ravel_logd = lambda p, x: logdens_fn(p, unravel(_gclip(x, grad_clipping)))
    logd_and_grad = jax.value_and_grad(ravel_logd, 1)

    def sample(key, params, state):
        gkey, ukey = jax.random.split(key)
        q1, g1, ld1 = state
        p1 = jax.random.normal(gkey, shape=q1.shape)
        potential_fn = lambda x: -ravel_logd(params, x)
        leapfrog = make_leapfrog(potential_fn, dt, round(length / dt), True)
        q2, p2, f2, v2 = leapfrog(q1, p1, -g1, -ld1)
        g2, ld2 = -f2, -v2
        ratio = (logd_gaussian(-p2).sum(-1)+ld2) - (logd_gaussian(p1).sum(-1)+ld1)
        (qn, gn, ldn), accepted = mh_select(ukey, ratio, state, (q2, g2, ld2))
        return (qn, gn, ldn), (unravel(qn), ldn)

    def init(key, params):
        sigma, mu = 1., 0.
        sample = jax.random.normal(key, (xsize,)) * sigma + mu
        return refresh((sample,), params)

    def refresh(state, params):
        sample = state[0]
        ld_new, grads_new = logd_and_grad(params, sample)
        return (sample, grads_new.conj(), ld_new)

    return MCSampler(sample, init, refresh)


def make_blackjax(logdens_fn, sample_shape, kernel="nuts", grad_clipping=None, **kwargs):
    from blackjax import hmc, nuts
    xsize, unravel = ravel_shape(sample_shape)
    inv_mass = 0.5 * jnp.ones(xsize)
    ravel_logd = lambda p, x: logdens_fn(p, unravel(_gclip(x, grad_clipping)))
    kmodule = {"hmc": hmc, "nuts": nuts}[kernel]

    def sample(key, params, state):
        logprob_fn = partial(ravel_logd, params)
        kernel = kmodule(logprob_fn, 
            inverse_mass_matrix=inv_mass, **kwargs)
        state = state[1]
        state, info = kernel.step(key, state)
        return (state.position, state), (unravel(state.position), -state.potential_energy)

    def refresh(state, params):
        sample = state[0]
        logprob_fn = partial(ravel_logd, params)
        return (sample, kmodule.init(sample, logprob_fn))

    init = _make_init_from_refresh(refresh, xsize)

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


def make_leapfrog(potential_fn, dt, steps, with_carry=True):
    pot_and_grad = jax.value_and_grad(potential_fn)

    def leapfrog_carry(q, p, g, v):
        # p for momentom and q for position
        # f for force and v for potential
        # simple Euler integration step
        def int_step(carry, _):
            q, p = carry
            q += dt * p
            p -= dt * pot_and_grad(q)[1]
            return (q, p), None
        # leapfrog by shifting half step
        p -= 0.5 * dt * g # first half p
        (q, p), _ = lax.scan(int_step, (q, p), None, length=steps-1)
        q += dt * p # final whole step update of q
        v, g = pot_and_grad(q) 
        p -= 0.5 * dt * g # final half p
        return q, p, g, v

    def leapfrog_nocarry(q, p):
        v, g = pot_and_grad(q)
        return leapfrog_carry(q, p, g, v)[:2]

    return leapfrog_carry if with_carry else leapfrog_nocarry


def _make_init_from_refresh(refresh_fn, size, mu=0., sigma=1.):
    def init(key, params):
        sample = jax.random.normal(key, (size,)) * sigma + mu
        return refresh_fn((sample,), params)
    return init


def _gclip(x, bnd):
    if bnd is None:
        return x
    if isinstance(bnd, (int, float, Array)):
        bnd = (-abs(bnd), abs(bnd))
    return clip_gradient(x, *bnd)