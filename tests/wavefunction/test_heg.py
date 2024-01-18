import jax
import numpy as np
import chex
import pytest
from jax import numpy as jnp

from fvmc.wavefunction.heg import (PlanewaveSlater, PairJastrowCCK,
                                   PairBackflow, IterativeBackflow,
                                   BackflowEtaKCM, BackflowEtaBessel)


_key0 = jax.random.PRNGKey(0)

_cell3d = jnp.eye(3) * 10.
_cell2d = jnp.array([[6., 0.], [2., 6.]])

_nelec = 16
_spins1 = [16]
_spins2 = [9, 7]


def _check_pbc(key, model, params, cell):
    # check pbc
    key1, key2 = jax.random.split(key)
    x = jax.random.uniform(key1, (_nelec, cell.shape[-1]))
    shift = jax.random.randint(key2, (_nelec, cell.shape[0],), -3, 3)
    x_shift = x + shift @ cell
    out1 = model.apply(params, x)
    out2 = model.apply(params, x_shift)
    chex.assert_trees_all_close(out2, out1)


def _check_perm(key, model, params, cell, anti_symm=True):
    # check permutation
    x = jax.random.uniform(key, (_nelec, cell.shape[-1]))
    perm = jnp.arange(_nelec, dtype=int).at[:2].set([1,0])
    x_perm = x[perm]
    sign1, logf1 = model.apply(params, x)
    sign2, logf2 = model.apply(params, x_perm)
    assert not jnp.iscomplexobj(logf1)
    np.testing.assert_allclose(sign2, -sign1 if anti_symm else sign1)
    np.testing.assert_allclose(logf2, logf1)


def _bf_jas(ret):
    if isinstance(ret, tuple):
        bf, jas = ret
    else:
        bf = ret
        jas = None
    return bf, jas


def _check_bf_pbc(key, model, params, cell):
    # check pbc, backflow covariant, jastrow invariant
    key1, key2 = jax.random.split(key)
    x = jax.random.uniform(key1, (_nelec, cell.shape[-1]))
    shift = jax.random.randint(key2, (_nelec, cell.shape[0],), -3, 3)
    x_shift = x + shift @ cell
    bf1, jas1 = _bf_jas(model.apply(params, x))
    bf2, jas2 = _bf_jas(model.apply(params, x_shift))
    np.testing.assert_allclose(bf2, bf1 + shift @ cell)
    if jas1 is not None:
      chex.assert_trees_all_close(jas1, jas2)


def _check_bf_perm(key, model, params, spins, cell):
    # check permutation, backflow covariant, jastrow invariant
    x = jax.random.uniform(key, (_nelec, cell.shape[-1]))
    perm = jnp.concatenate(jax.tree_map(
        jax.random.permutation,
        list(jax.random.split(key, len(spins))),
        jnp.split(jnp.arange(sum(spins)), np.cumsum(spins)[:-1])))
    x_perm = x[perm]
    bf1, jas1 = _bf_jas(model.apply(params, x))
    bf2, jas2 = _bf_jas(model.apply(params, x_perm))
    np.testing.assert_allclose(bf2, bf1[perm])
    if jas1 is not None:
      chex.assert_trees_all_close(jas1, jas2)


@pytest.mark.parametrize("full_det", [True, False])
@pytest.mark.parametrize("spin_symmetry", [True, False])
@pytest.mark.parametrize("multi_det", [None, 4])
@pytest.mark.parametrize("cell", [_cell3d, _cell2d])
@pytest.mark.parametrize("spins", [_spins1, _spins2])
def test_planewave_slater(spins, cell, multi_det, spin_symmetry, full_det):
    model = PlanewaveSlater(spins=spins, cell=cell, multi_det=multi_det,
                            spin_symmetry=spin_symmetry, full_det=full_det)
    params = model.init(_key0, jnp.zeros((_nelec, cell.shape[-1])))
    _check_pbc(_key0, model, params, cell)
    _check_perm(_key0, model, params, cell, anti_symm=True)


@pytest.mark.parametrize("cell", [_cell3d, _cell2d])
@pytest.mark.parametrize("spins", [_spins1, _spins2])
def test_pair_jastrow_cck(spins, cell):
    model = PairJastrowCCK(spins=spins, cell=cell)
    params = model.init(_key0, jnp.zeros((_nelec, cell.shape[-1])))
    _check_pbc(_key0, model, params, cell)
    _check_perm(_key0, model, params, cell, anti_symm=False)


@pytest.mark.parametrize("eta_name", ['kcm', 'bessel'])
@pytest.mark.parametrize("cell", [_cell3d, _cell2d])
@pytest.mark.parametrize("spins", [_spins1, _spins2])
def test_pair_backflow(spins, cell, eta_name):
    if eta_name == 'kcm':
        eta = BackflowEtaKCM()
    elif eta_name == 'bessel':
        nbasis = 5
        rcut = cell[0, 0]/2.0
        eta = BackflowEtaBessel(nbasis, rcut)
    model = PairBackflow(spins, cell, eta)
    params = model.init(_key0, jnp.zeros((_nelec, cell.shape[-1])))
    _check_bf_pbc(_key0, model, params, cell)
    _check_bf_perm(_key0, model, params, spins, cell)


@pytest.mark.parametrize("nlayer", [1, 2, 3])
@pytest.mark.parametrize("cell", [_cell3d, _cell2d])
@pytest.mark.parametrize("spins", [_spins1, _spins2])
def test_iter_backflow(spins, cell, nlayer):
    model = IterativeBackflow(spins, cell, [BackflowEtaKCM() for _ in range(nlayer)])
    params = model.init(_key0, jnp.zeros((_nelec, cell.shape[-1])))
    _check_bf_pbc(_key0, model, params, cell)
    _check_bf_perm(_key0, model, params, spins, cell)
