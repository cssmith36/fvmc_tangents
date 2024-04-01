import jax
import numpy as np
import pytest
from jax import numpy as jnp

from fvmc.wavefunction.neuralnet import FermiNet
from fvmc.utils import parse_spin_num
from .test_naive import make_collapse_conf


_key0 = jax.random.PRNGKey(0)


_conf1 = {"full_det": False,
          "fermilayer": {
              "h2_convolution": False,
              "identical_h1_update": False,
              "identical_h2_update": False,
              "spin_symmetry": False}}
_conf2 = {"full_det": False,
          "fermilayer": {
              "h2_convolution": False,
              "identical_h1_update": True,
              "identical_h2_update": False,
              "spin_symmetry": True}}
_conf3 = {"full_det": True,
          "fermilayer": {
              "h2_convolution": True,
              "identical_h1_update": False,
              "identical_h2_update": True,
              "spin_symmetry": True}}


@pytest.fixture(scope='module', params=[_conf1, _conf2, _conf3])
def model_data(request):
    r, elems, x = make_collapse_conf()
    x = x[0]
    x = x + jax.random.normal(_key0, x.shape)
    elems = jnp.sort(elems)
    n_el = x.shape[0]
    spins = parse_spin_num(n_el, None)
    model = FermiNet(elems=elems, spins=spins, **request.param)
    params = model.init(_key0, r, x)
    return model, params, r, x


@pytest.mark.slow
def test_elec_antisymm(model_data):
    model, params, r, x = model_data
    iperm = jnp.arange(x.shape[0], dtype=int).at[:2].set([1,0])
    px = x[iperm, :]

    sign1, logf1 = model.apply(params, r, x)
    sign2, logf2 = model.apply(params, r, px)
    np.testing.assert_allclose(sign1, -sign2, equal_nan=False)
    np.testing.assert_allclose(logf1, logf2, equal_nan=False)


@pytest.mark.slow
def test_nucl_symm(model_data):
    model, params, r, x = model_data
    iperm = jnp.arange(r.shape[0], dtype=int).at[:2].set([1,0])
    pr = r[iperm, :]

    sign1, logf1 = model.apply(params, r, x)
    sign2, logf2 = model.apply(params, pr, x)
    np.testing.assert_allclose(sign1, sign2, equal_nan=False)
    np.testing.assert_allclose(logf1, logf2, equal_nan=False)


# @pytest.mark.slow
# @pytest.mark.parametrize("config", [_conf2, _conf3])
# def test_spin_symm(config):
#     r, elems, x = make_collapse_conf()
#     x = x[0, :-1] # 8 electrons
#     elems = jnp.sort(elems)
#     n_el = x.shape[0]
#     n_up, n_dn = parse_spin_num(n_el, None)
#     model = FermiNet(elems=elems, **config)
#     params = model.init(_key0, r, x)

#     x = x + jax.random.normal(_key0, x.shape)
#     px = jnp.concatenate([x[n_up:], x[:n_up]], axis=0)

#     sign1, logf1 = model.apply(params, r, x)
#     sign2, logf2 = model.apply(params, r, px)
#     # np.testing.assert_allclose(sign1, sign2, equal_nan=False)
#     np.testing.assert_allclose(logf1, logf2, equal_nan=False)
