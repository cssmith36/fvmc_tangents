#!/usr/bin/env python3
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from fvmc.moire import OneShell
from fvmc.utils import replicate_cell
from fvmc.estimator import build_eval_local_elec

from .test_estimator import get_sign_log, make_dummy_model, make_test_log_f


def _get_tri_conf(rs, nx):
    am = (2*np.pi/3**0.5)**0.5*rs
    cell = am*np.array([
        [1, 0],
        [-0.5, 3**0.5/2],
    ])
    ndim = len(cell)
    pos = np.zeros([1, ndim])
    pos, cell = replicate_cell(pos, cell, (nx,)*ndim)
    return cell, pos


def _get_rect_conf(rs, nx):
    am = (2*np.pi/3**0.5)**0.5*rs
    cell = am*np.array([
        [1, 0],
        [0, 3**0.5],
    ])
    ndim = len(cell)
    pos = np.array([
        [0, 0],
        [0.5, 0.5],
    ]) @ cell
    pos, cell = replicate_cell(pos, cell, (nx,)*ndim)
    return cell, pos


def _get_moire_special_points(rs):
    am = (2*np.pi/3**0.5)**0.5*rs
    cell = am*np.array([
        [1, 0],
        [-0.5, 3**0.5/2],
    ])
    fracs = np.array([
        [0, 0],
        [1./3, 2./3],
        [2./3, 1./3],
    ])
    pos = fracs@cell
    return pos


test_cases = [
    (0, [-6, 3, 3]),  # triangular lattice
    (np.pi/4, [-4.24264069, -1.55291427, 5.79555496]),
    (np.pi/3, [-3, -3, 6]),  # honeycomb lattice
]


@pytest.mark.parametrize("test_case", test_cases)
@pytest.mark.parametrize("conf_fn", [_get_tri_conf, _get_rect_conf])
def test_moire_one_shell(conf_fn, test_case):
    nx = 1
    rs = 5.0
    am = (2*np.pi/3**0.5)**0.5*rs
    cell, pos = conf_fn(rs, nx)
    testpos = _get_moire_special_points(rs)

    vm = -1.0
    phi, res = test_case
    pot = OneShell(cell, am, vm, phi)
    vals = pot(testpos)
    np.testing.assert_allclose(vals, res)


@pytest.mark.parametrize("nx", [1, 2, 3])
@pytest.mark.parametrize("conf_fn", [_get_tri_conf, _get_rect_conf])
def test_moire_incommensurate(conf_fn, nx):
    rs = 5.0
    am = (2*np.pi/3**0.5)**0.5*rs
    cell, pos = conf_fn(rs, nx)

    vm = -1.0
    phi = 0.0
    with pytest.raises(RuntimeError):
        OneShell(cell, 1.01*am, vm, phi)


def test_moire_local_energy():
    # moire related
    nx = 1
    rs = 5.0
    am = (2*np.pi/3**0.5)**0.5*rs
    vm = -1.0
    phi, res = test_cases[0]
    # dummy system
    f, logf = make_test_log_f()
    model = make_dummy_model(get_sign_log(f))
    cell, pos = _get_tri_conf(rs, nx)
    testpos = _get_moire_special_points(rs)
    nuclei = jnp.zeros((0, pos.shape[-1]), dtype=float)
    elems = jnp.zeros((0,), dtype=int)
    local_fn = build_eval_local_elec(
        model, elems, nuclei, cell,
        ext_pots={"moire": {"am_length": am, "vm_depth": vm, "phi_shape": phi}})
    eloc, sign, logf, extras = local_fn(None, testpos)
    np.testing.assert_allclose(extras['e_moire'], np.sum(res))
