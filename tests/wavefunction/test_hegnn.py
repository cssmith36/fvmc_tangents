import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp

from fvmc.wavefunction.heg import (ElecProductModel, PairJastrowCCK,
                                   PlanewaveSlater)
from fvmc.wavefunction.hegnn import NeuralBackflow

from .test_heg import _cell2d, _cell3d, _key0, _nelec, _spins1, _spins2
from .test_heg import _check_pbc, _check_perm, _check_bf_pbc, _check_bf_perm


@pytest.mark.slow
@pytest.mark.parametrize("cell", [_cell3d, _cell2d])
@pytest.mark.parametrize("spins", [_spins1, _spins2])
def test_neural_backflow(spins, cell):
    model = NeuralBackflow(spins=spins, cell=cell)
    params = model.init(_key0, jnp.zeros((_nelec, cell.shape[-1])))
    _check_bf_pbc(_key0, model, params, cell)
    _check_bf_perm(_key0, model, params, spins, cell)


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
