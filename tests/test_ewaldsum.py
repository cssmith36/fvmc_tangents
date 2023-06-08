# many test functions are borrowed from pyqmc

import jax
import pytest
import numpy as np
from jax import numpy as jnp

from fvmc.ewaldsum import EwaldSum


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
    shift_vec = jax.random.normal(_key0, shape=(100, 1, 3))
    shifted_pos = pos + shift_vec # [n_batch, n_particle, 3]
    ewald = EwaldSum(latvec)
    e_raw = ewald.energy(charge, pos)
    e_shifted = jax.vmap(ewald.energy, in_axes=(None, 0))(charge, shifted_pos)
    np.testing.assert_allclose(*jnp.broadcast_arrays(e_raw, e_shifted))