import jax
import numpy as np
import pytest
from jax import numpy as jnp

from fvmc.sampler import (build_conf_init_fn, choose_sampler_builder,
                          make_batched, make_chained, make_multistep)

_mean = 0.5
_std = 0.5
_logprob_fn = lambda p, x: jnp.sum(-0.5*((x - _mean) / _std)**2, -1)
_xshape = 1

_nchain = 50
_nstep = 200
_nburn = 100

_key0 = jax.random.PRNGKey(0)


def make_test_sampler(name):
    maker = choose_sampler_builder(name)
    if name == "gaussian":
        sampler = maker(_logprob_fn, _xshape, mu=_mean, sigma=_std)
    elif name == "black":
        sampler = maker(_logprob_fn, _xshape, step_size=0.1, kernel="hmc", num_integration_steps=10)
    else:
        sampler = maker(_logprob_fn, _xshape)
    return make_multistep(make_batched(sampler, _nchain, concat=False), _nstep, concat=False)


def shared_sampler_test(sampler, jit=True):
    params = None
    key1, key2, key3 = jax.random.split(_key0, 3)

    if jit:
        sampler = jax.tree_map(jax.jit, sampler)

    state = sampler.init(key1, params)
    state = sampler.burn_in(key2, params, state, _nburn)
    state, (sample, logprob), info = sampler.sample(key3, params, state)
    state = sampler.refresh(sample, params)

    assert sample.shape == (_nstep, _nchain, _xshape)
    assert logprob.shape == (_nstep, _nchain)
    np.testing.assert_allclose(logprob, _logprob_fn(None, sample), atol=1e-5)
    np.testing.assert_allclose(sample.mean(), _mean, rtol=0.05)
    np.testing.assert_allclose(sample.std(), _std, rtol=0.05)


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
    sampler = make_test_sampler(name)
    shared_sampler_test(sampler, jit=True)


@pytest.mark.slow
def test_sampler_hmc_jittered():
    sampler = choose_sampler_builder("hmc")(_logprob_fn, _xshape, jittered=True)
    sampler = make_multistep(make_batched(sampler, _nchain, concat=False), _nstep, concat=False)
    shared_sampler_test(sampler, jit=True)


@pytest.mark.slow
def test_sampler_chained():
    mcmc = make_test_sampler("mcmc")
    mala = make_test_sampler("mala")
    sampler = make_chained(mcmc, mala)
    shared_sampler_test(sampler, jit=True)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["hmc", "mala"])
def test_sampler_grad_clipping(name):
    maker = choose_sampler_builder(name)
    sampler = maker(_logprob_fn, _xshape, grad_clipping=0.5)
    sampler = make_multistep(make_batched(sampler, _nchain, concat=False), _nstep, concat=False)
    shared_sampler_test(sampler, jit=True)
