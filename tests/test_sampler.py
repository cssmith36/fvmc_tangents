import operator as _op

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from fvmc.utils import exp_shifted
from fvmc.sampler import (build_conf_init_fn, choose_adaptive_builder,
                          choose_sampler_builder, make_batched, make_chained,
                          make_multistep, relavistic_ke, sample_relavistic_momentum)

_mean = 0.5
_std = 0.5
_logprob_fn = lambda p, x: jnp.sum(-0.5*((x - _mean) / _std)**2, -1)
_xshape = 2

_nchain = 50
_nstep = 200
_nburn = 100

_key0 = jax.random.PRNGKey(0)


def make_test_sampler(name, adaptive=False):
    maker = (choose_sampler_builder(name) if not adaptive else
             choose_adaptive_builder(name, harmonic=(name == "hmc"), interval=100))
    if name == "gaussian":
        sampler = maker(_logprob_fn, _xshape, mu=_mean, sigma=_std)
    elif name == "black":
        sampler = maker(_logprob_fn, _xshape, step_size=0.1, kernel="hmc", num_integration_steps=10)
    else:
        sampler = maker(_logprob_fn, _xshape, mass=jnp.ones(_xshape))
    return make_multistep(make_batched(sampler, _nchain, concat=False), _nstep, concat=False)


def shared_sampler_test(sampler, jit=True, check_info=True):
    params = None
    key1, key2, key3 = jax.random.split(_key0, 3)

    if jit:
        sampler = jax.tree_map(jax.jit, sampler)

    state = sampler.init(key1, params)
    state = sampler.burn_in(key2, params, state, _nburn)
    state, (sample, logprob), info = sampler.sample(key3, params, state)
    state = sampler.refresh(state, params)

    assert sample.shape == (_nstep, _nchain, _xshape)
    assert logprob.shape == (_nstep, _nchain)
    np.testing.assert_allclose(logprob, _logprob_fn(None, sample), atol=1e-5)
    np.testing.assert_allclose(sample.mean(), _mean, rtol=0.05)
    np.testing.assert_allclose(sample.std(), _std, rtol=0.05)

    if check_info:
        np.testing.assert_array_compare(_op.le, info["is_accepted"], 1)
        np.testing.assert_array_compare(_op.ge, info["recip_ratio"], 1)


@pytest.mark.parametrize("with_r", [True, False])
def test_conf_init_shape(with_r):
    elems = jnp.array([2, 2])
    nuclei = jnp.array([[-1, 0, 0], [1., 0, 0]])
    for n_elec in (3, 4, 5):
        init_fn = build_conf_init_fn(elems, nuclei, n_elec, with_r=with_r)
        if with_r:
            init_r, init_x = init_fn(_key0)
            assert init_r.shape == nuclei.shape
        else:
            init_x = init_fn(_key0)
        assert init_x.shape == (n_elec, nuclei.shape[-1])


def test_conf_init_collapse():
    elems = jnp.array([2, 2])
    nuclei = jnp.array([[-1, 0, 0], [1., 0, 0]])
    n_elec = 4
    init_fn = build_conf_init_fn(elems, nuclei, n_elec, with_r=False, sigma_x=0)
    init_x = init_fn(_key0)
    np.testing.assert_allclose(init_x, jnp.concatenate([nuclei, nuclei]))


def test_sampler_gaussian():
    sampler = make_test_sampler("gaussian")
    shared_sampler_test(sampler, jit=True)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["mcmc", "mala", "hmc", "black"])
def test_sampler_distribution(name):
    sampler = make_test_sampler(name, adaptive=False)
    shared_sampler_test(sampler, jit=True)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["mcmc", "mala", "hmc"])
def test_sampler_adaptive(name):
    sampler = make_test_sampler(name, adaptive=True)
    shared_sampler_test(sampler, jit=True)


@pytest.mark.slow
def test_sampler_hmc_jittered():
    sampler = choose_sampler_builder("hmc")(_logprob_fn, _xshape, jitter_dt=0.5)
    sampler = make_multistep(make_batched(sampler, _nchain, concat=False), _nstep, concat=False)
    shared_sampler_test(sampler, jit=True)


@pytest.mark.slow
def test_sampler_chained():
    mcmc = make_test_sampler("mcmc")
    mala = make_test_sampler("mala")
    sampler = make_chained(mcmc, mala)
    shared_sampler_test(sampler, jit=True, check_info=False)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["hmc", "mala"])
def test_sampler_grad_clipping(name):
    maker = choose_sampler_builder(name)
    sampler = maker(_logprob_fn, _xshape, grad_clipping=0.5)
    sampler = make_multistep(make_batched(sampler, _nchain, concat=False), _nstep, concat=False)
    shared_sampler_test(sampler, jit=True)


@pytest.mark.slow
@pytest.mark.parametrize("c", [1, 10])
def test_relativistic_kinetic(c):
    p = sample_relavistic_momentum(_key0, c, shape=(2000000,))
    # check velocity is bounded
    v = jax.jit(jax.vmap(jax.grad(lambda p: relavistic_ke(p, c=c))))(p)
    assert jnp.all((v < c) & (v > -c))
    # check momentum distribution is correct
    lim = abs(p).max() * 1.1
    counts, bins = jnp.histogram(p, bins=100, range=(-lim, lim), density=True)
    mid = (bins[1:] + bins[:-1]) / 2
    ke = jax.jit(relavistic_ke)(mid, c=c)
    density = exp_shifted(-ke, normalize="sum")[0] / (2 * lim / counts.shape[0])
    np.testing.assert_allclose(counts, density, atol=3e-3)


@pytest.mark.slow
@pytest.mark.parametrize("speed_limit", [1., 10.])
def test_sampler_hmc_relativistic(speed_limit):
    sampler = choose_sampler_builder("hmc")(
        _logprob_fn, _xshape, length=None, steps=10, segments=2, speed_limit=speed_limit)
    sampler = make_multistep(make_batched(sampler, _nchain, concat=False), _nstep, concat=False)
    shared_sampler_test(sampler, jit=True)
