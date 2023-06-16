import jax
import numpy as np
import chex
import pytest
from jax import numpy as jnp

from fvmc.neuralnet_pbc import FermiNetPbc, raw_features_pbc
from fvmc.utils import parse_spin
from .test_wavefunction import make_collapse_conf


_key0 = jax.random.PRNGKey(0)


_conf1 = {"identical_h2_update": False, "spin_symmetry": False}
_conf2 = {"identical_h2_update": True, "spin_symmetry": True}


def test_feature_pbc():
    n_freq = 5
    r, elems, x = make_collapse_conf()
    cell = jnp.eye(3) * 6.
    x = x[0]
    x = x + jax.random.normal(_key0, x.shape)

    px = x.at[0, 0].add(6.)
    pr = r.at[0, 0].add(6.)

    res1 = raw_features_pbc(r, x, cell, n_freq)
    res2 = raw_features_pbc(pr, x, cell, n_freq)
    res3 = raw_features_pbc(r, px, cell, n_freq)

    chex.assert_tree_all_close(res1, res2, atol=1e-10)
    chex.assert_tree_all_close(res1, res3, atol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("config", [_conf1, _conf2])
def test_elec_antisymm(config):
    r, elems, x = make_collapse_conf()
    cell = jnp.eye(3) * 6.
    x = x[0]
    elems = jnp.sort(elems)
    n_el = x.shape[0]
    spins = parse_spin(n_el, None)
    model = FermiNetPbc(elems=elems, spins=spins, cell=cell, **config)
    params = model.init(_key0, r, x)

    x = x + jax.random.normal(_key0, x.shape)
    iperm = jnp.arange(n_el, dtype=int).at[:2].set([1,0])
    px = x[iperm, :]
    
    sign1, logf1 = model.apply(params, r, x)
    sign2, logf2 = model.apply(params, r, px)
    np.testing.assert_allclose(sign1, -sign2, equal_nan=False)
    np.testing.assert_allclose(logf1, logf2, equal_nan=False)


@pytest.mark.slow
@pytest.mark.parametrize("config", [_conf1, _conf2])
def test_nucl_symm(config):
    r, elems, x = make_collapse_conf()
    cell = jnp.eye(3) * 6.
    x = x[0]
    elems = jnp.sort(elems)
    n_nucl = r.shape[0]
    n_el = x.shape[0]
    spins = parse_spin(n_el, None)
    model = FermiNetPbc(elems=elems, spins=spins, cell=cell, **config)
    params = model.init(_key0, r, x)

    x = x + jax.random.normal(_key0, x.shape)
    iperm = jnp.arange(n_nucl, dtype=int).at[:2].set([1,0])
    pr = r[iperm, :]
    
    sign1, logf1 = model.apply(params, r, x)
    sign2, logf2 = model.apply(params, pr, x)
    np.testing.assert_allclose(sign1, sign2, equal_nan=False)
    np.testing.assert_allclose(logf1, logf2, equal_nan=False)


@pytest.mark.slow
@pytest.mark.parametrize("config", [_conf1, _conf2])
def test_particle_pbc(config):
    r, elems, x = make_collapse_conf()
    cell = jnp.eye(3) * 6.
    x = x[0]
    elems = jnp.sort(elems)
    n_nucl = r.shape[0]
    n_el = x.shape[0]
    spins = parse_spin(n_el, None)
    model = FermiNetPbc(elems=elems, spins=spins, cell=cell, **config)
    params = model.init(_key0, r, x)

    x = x + jax.random.normal(_key0, x.shape)
    px = x.at[0, 0].add(6.)
    pr = r.at[0, 0].add(6.)
    
    sign1, logf1 = model.apply(params, r, x)
    sign2, logf2 = model.apply(params, pr, x)
    sign3, logf3 = model.apply(params, r, px)

    np.testing.assert_allclose(sign1, sign2, equal_nan=False)
    np.testing.assert_allclose(sign1, sign3, equal_nan=False)
    np.testing.assert_allclose(logf1, logf2, equal_nan=False)
    np.testing.assert_allclose(logf1, logf3, equal_nan=False)
