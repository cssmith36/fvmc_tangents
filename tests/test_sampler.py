import pytest
import jax
import numpy as np
from jax import numpy as jnp

from vdmc.sampler import choose_sampler_maker, make_batched, make_multistep


_mean = 0.5
_std = 0.5
_logprob_fn = lambda p, x: jnp.sum(-0.5*((x - _mean) / _std)**2, -1)
_xshape = 1

_nchain = 50
_nstep = 200
_nburn = 100

_key0 = jax.random.PRNGKey(0)


def make_test_sampler(name):
    maker = choose_sampler_maker(name)
    if name == "gaussian":
        sampler = maker(_logprob_fn, _xshape, mu=_mean, sigma=_std)
    elif name == "black":
        sampler = maker(_logprob_fn, _xshape, step_size=0.1, kernel="hmc", num_integration_steps=10)
    else:
        sampler = maker(_logprob_fn, _xshape)
    return make_multistep(make_batched(sampler, _nchain, concat=False), _nstep, concat=False)
    

# @pytest.mark.slow
@pytest.mark.parametrize("name", ["gaussian", "mcmc", "mala", "hmc", "black"])
def test_sampler_gaussian(name):
    sampler = make_test_sampler(name)
    params = None
    key1, key2, key3 = jax.random.split(_key0, 3)

    state = sampler.init(key1, params)
    state = sampler.burn_in(key2, params, state, _nburn)
    state, (sample, logprob) = jax.jit(sampler.sample)(key3, params, state)
    state = sampler.refresh(sample, params)

    assert sample.shape == (_nstep, _nchain, _xshape)
    assert logprob.shape == (_nstep, _nchain)
    np.testing.assert_allclose(logprob, _logprob_fn(None, sample))
    np.testing.assert_allclose(sample.mean(), _mean, rtol=0.05)
    np.testing.assert_allclose(sample.std(), _std, rtol=0.05)