import jax
import numpy as np
import chex
import pytest
from jax import numpy as jnp

from fvmc.wavefunction.heg import ElecProductModel, PlanewaveSlater, PairJastrowCCK


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
    chex.assert_tree_all_close(out2, out1)


def _check_perm(key, model, params, cell, anti_symm=True):
    # check permutation
    x = jax.random.uniform(key, (_nelec, cell.shape[-1]))
    perm = jnp.arange(_nelec, dtype=int).at[:2].set([1,0])
    x_perm = x[perm]
    sign1, logf1 = model.apply(params, x)
    sign2, logf2 = model.apply(params, x_perm)
    np.testing.assert_allclose(sign2, -sign1 if anti_symm else sign1)
    np.testing.assert_allclose(logf2, logf1)


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

