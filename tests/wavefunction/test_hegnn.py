import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp

from fvmc.wavefunction.heg import (ElecProductModel, PairJastrowCCK,
                                   PlanewaveSlater)
from fvmc.wavefunction.hegnn import NeuralBackflow

from .test_heg import _cell2d, _cell3d, _key0, _nelec, _spins1, _spins2
from .test_heg import _check_pbc, _check_perm


def _check_nnbf_pbc(key, model, params, cell):
    # check pbc, backflow covariant, jastrow invariant
    key1, key2 = jax.random.split(key)
    x = jax.random.uniform(key1, (_nelec, cell.shape[-1]))
    shift = jax.random.randint(key2, (_nelec, cell.shape[0],), -3, 3)
    x_shift = x + shift @ cell
    bf1, jas1 = model.apply(params, x)
    bf2, jas2 = model.apply(params, x_shift)
    np.testing.assert_allclose(bf2, bf1 + shift @ cell)
    chex.assert_trees_all_close(jas1, jas2)


def _check_nnbf_perm(key, model, params, spins, cell):
    # check permutation, backflow covariant, jastrow invariant
    x = jax.random.uniform(key, (_nelec, cell.shape[-1]))
    perm = jnp.concatenate(jax.tree_map(
        jax.random.permutation,
        list(jax.random.split(key, len(spins))),
        jnp.split(jnp.arange(sum(spins)), np.cumsum(spins)[:-1])))
    x_perm = x[perm]
    bf1, jas1 = model.apply(params, x)
    bf2, jas2 = model.apply(params, x_perm)
    np.testing.assert_allclose(bf2, bf1[perm])
    chex.assert_trees_all_close(jas1, jas2)


@pytest.mark.slow
@pytest.mark.parametrize("cell", [_cell3d, _cell2d])
@pytest.mark.parametrize("spins", [_spins1, _spins2])
def test_neural_backflow(spins, cell):
    model = NeuralBackflow(spins=spins, cell=cell)
    params = model.init(_key0, jnp.zeros((_nelec, cell.shape[-1])))
    _check_nnbf_pbc(_key0, model, params, cell)
    _check_nnbf_perm(_key0, model, params, spins, cell)


@pytest.mark.slow
@pytest.mark.parametrize("cell", [_cell3d, _cell2d])
@pytest.mark.parametrize("spins", [_spins1, _spins2])
@pytest.mark.parametrize("apply_bf", [True, [True, False]])
def test_productmodel_nnbf(spins, cell, apply_bf):
    model = ElecProductModel(
        submodels=[PlanewaveSlater(spins=spins, cell=cell),
                   PairJastrowCCK(spins=spins, cell=cell)],
        backflow=NeuralBackflow(spins=spins, cell=cell),
        apply_backflow=apply_bf)
    params = model.init(_key0, jnp.zeros((_nelec, cell.shape[-1])))
    _check_pbc(_key0, model, params, cell)
    _check_perm(_key0, model, params, cell, anti_symm=True)
