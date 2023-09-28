# many test functions are borrowed from pyqmc

import jax
import pytest
import numpy as np
from jax import numpy as jnp

from fvmc.ewaldsum import EwaldSum
from fvmc.ewaldsum import determine_cell_type, gen_pbc_disp_fn
from fvmc.utils import displace_matrix


_key0 = jax.random.PRNGKey(0)


def _get_NaCl_conf():
    L = 2.0
    latvec = (jnp.ones((3, 3)) - np.eye(3)) * L / 2
    charge = jnp.array([1, -1])
    pos = jnp.array([
        [0., 0., 0.],
        [L/2, L/2, L/2]
    ])
    answer = -1.74756
    return latvec, charge, pos, answer


def _get_NaCl_conf1():
    L = 2.0
    latvec = jnp.eye(3) * L
    charge = jnp.array([1, 1, 1, 1, -1, -1, -1, -1])
    r = jnp.array([
        [0., 0., 0.],
        [0., L/2, L/2],
        [L/2, 0., L/2],
        [L/2, L/2, 0.]
    ])
    x = L/2 - r
    pos = jnp.concatenate([r, x], axis=0)
    answer = -1.74756 * 4
    return latvec, charge, pos, answer


def _get_CaF2_conf():
    L = 4 / np.sqrt(3)
    latvec = (jnp.ones((3, 3)) - np.eye(3)) * L / 2
    charge = jnp.array([2, -1, -1])
    r = jnp.array([[0, 0, 0.]])
    x = jnp.full((2, 3),  L / 4).at[1, 1].multiply(-1)
    pos = jnp.concatenate([r, x], axis=0)
    answer = -5.03879
    return latvec, charge, pos, answer


@pytest.mark.slow
@pytest.mark.parametrize("conf", [_get_NaCl_conf, _get_NaCl_conf1, _get_CaF2_conf])
def test_ewaldsum_simple(conf):
    latvec, charge, pos, answer = conf()
    ewald = EwaldSum(latvec)
    np.testing.assert_allclose(ewald.energy(charge, pos), answer, rtol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("conf", [_get_NaCl_conf, _get_NaCl_conf1, _get_CaF2_conf])
def test_ewaldsum_transinv(conf):
    latvec, charge, pos, answer = conf()
    shift_vec = jax.random.normal(_key0, shape=(100, 1, 3)) * 5.
    shifted_pos = pos + shift_vec # [n_batch, n_particle, 3]
    ewald = EwaldSum(latvec)
    e_raw = ewald.energy(charge, pos)
    e_shifted = jax.vmap(ewald.energy, in_axes=(None, 0))(charge, shifted_pos)
    np.testing.assert_allclose(*jnp.broadcast_arrays(e_raw, e_shifted), rtol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("conf", [_get_NaCl_conf, _get_CaF2_conf])
def test_ewaldsum_replicate(conf):
    from fvmc.utils import replicate_cell
    latvec, charge, pos, answer = conf()
    e_raw = EwaldSum(latvec).energy(charge, pos)
    rpos, rlatvec = replicate_cell(pos, latvec, (2, 2, 2))
    rcharge = jnp.tile(charge, 8)
    e_rep = EwaldSum(rlatvec).energy(rcharge, rpos)
    np.testing.assert_allclose(e_rep, e_raw * 8, rtol=1e-7)


@pytest.mark.slow
def test_ewaldsum_calc_pe():
    latvec, charge, pos, answer = _get_NaCl_conf1()
    elems = charge[:4]
    r, x = jnp.split(pos, [4], axis=0)
    ewald = EwaldSum(latvec)
    np.testing.assert_allclose(ewald.calc_pe(elems, r, x), answer, rtol=1e-5)


# below are tests for pbc distance

_cell_dict = dict(
        diagonal = jnp.eye(3),
        orthogonal = jnp.eye(3, k=1) + jnp.eye(3, k=-2),
        general = jnp.ones((3,3)) - jnp.eye(3)
)


@pytest.mark.parametrize("mode", ["diagonal", "orthogonal", "general"])
def test_determine_cell_type(mode):
    cell = _cell_dict[mode]
    assert determine_cell_type(cell) == mode


@pytest.mark.parametrize("cell_type", ["diagonal", "orthogonal", "general"])
def test_pbc_displacement(cell_type):
    npoints = 100
    keya, keyb = jax.random.split(_key0)
    xa = jax.random.uniform(keya, (npoints, 3))
    xb = jax.random.uniform(keyb, (npoints, 3))
    cell = _cell_dict[cell_type]
    cell_types = list(_cell_dict.keys())
    self_idx = cell_types.index(cell_type)

    auto_disp_fn = gen_pbc_disp_fn(cell, mode='auto')
    ref_dmat = displace_matrix(xa, xb, auto_disp_fn)
    neg_transpose_dmat = displace_matrix(xb, xa, auto_disp_fn)
    np.testing.assert_allclose(-neg_transpose_dmat.transpose((1,0,2)), ref_dmat)

    for idx, mode in enumerate(cell_types):
        disp_fn = gen_pbc_disp_fn(cell, mode)
        dmat = displace_matrix(xa, xb, disp_fn=disp_fn)
        if idx < self_idx: # simple method, should not work
            with np.testing.assert_raises(AssertionError):
                np.testing.assert_allclose(dmat, ref_dmat, atol=0.01)
        else: # general method, should work
            np.testing.assert_allclose(dmat, ref_dmat, atol=1e-10)
