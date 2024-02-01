# many test functions are borrowed from vmcnet

from functools import partial

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from fvmc.hamiltonian import (calc_ke_elec, calc_ke_full, calc_local_energy,
                              calc_pe, laplacian_over_f)
from fvmc.utils import split_spin


def make_test_log_f():

    def f(params, x):
        del params
        return jnp.sum(jnp.square(x) + 3 * x, axis=(-1,-2))

    def log_f(params, x):
        return jnp.log(jnp.abs(f(params, x)))

    return f, log_f


def make_test_x():
    return jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def make_batched_x():
    return jnp.array(
        [
            [[0.0, -1.0], [0.0, 1.0]],
            [[2.0, 3.0], [-3.0, 0.0]],
        ]
    )


def make_test_ions():
    ion_pos = jnp.array([[-4.0, 0.0], [0.0, 0.0], [2.0, 1.0]])
    ion_charges = jnp.array([1.0, 2.0, 3.0])
    return ion_pos, ion_charges


def test_potential_energy():
    x = make_batched_x()
    nuclei, elems = make_test_ions()

    target_el_ion = -jnp.sum(jnp.array([
            [
                [1.0 / jnp.sqrt(17.0), 2.0 / 1.0, 3.0 / jnp.sqrt(8.0)],
                [1.0 / jnp.sqrt(17.0), 2.0 / 1.0, 3.0 / 2.0],
            ],
            [
                [1.0 / jnp.sqrt(45.0), 2.0 / jnp.sqrt(13.0), 3.0 / 2.0],
                [1.0 / 1.0, 2.0 / 3.0, 3.0 / jnp.sqrt(26.0)],
            ],
        ]), axis=(-1,-2))
    target_el_el = jnp.array([1.0 / 2.0, 1.0 / jnp.sqrt(34.0)])
    target_ion_ion = (2.0 / 4.0) + (3.0 / jnp.sqrt(37.0)) + (6.0 / jnp.sqrt(5.0))

    target_pe = target_el_ion + target_el_el + target_ion_ion
    actual_pe = calc_pe(elems, nuclei, x)

    np.testing.assert_allclose(actual_pe, target_pe)


@pytest.mark.parametrize("partition_size", [None, 1, 2])
@pytest.mark.parametrize("forward_mode", [True, False])
@pytest.mark.parametrize("scale", [1., 0.5])
def test_laplacian(scale, forward_mode, partition_size):
    x = make_test_x()
    f, log_f = make_test_log_f()
    log_psi = partial(log_f, None)

    # d^2f/dx_i^2 = 2 for all i, so the Laplacian is simply 2 * n, where n is
    # the number of coordiantes. We then divide by f(x) to get (nabla^2 f) / f
    target_laplacian = 2 * scale * (x.shape[-1] * x.shape[-2]) / f(None, x)
    actual_laplacian = laplacian_over_f(
        log_psi,
        scale=scale,
        forward_mode=forward_mode,
        partition_size=partition_size)(x)

    np.testing.assert_allclose(actual_laplacian, target_laplacian, rtol=1e-6)


@pytest.mark.parametrize("x", [make_test_x(), make_batched_x()])
def test_kinetic_energy(x):
    # single x
    """Test (nabla^2 f)(x) / f(x) for f(x) = sum_i x_i^2 + 3x_i."""
    f, log_f = make_test_log_f()
    log_psi = partial(log_f, None)

    # d^2f/dx_i^2 = 2 for all i, so the Laplacian is simply 2 * n, where n is
    # the number of coordiantes. We then divide by f(x) to get (nabla^2 f) / f
    target_ke = -0.5 * (x.shape[-1] * x.shape[-2]) * 2 / f(None, x)
    ke_fn = partial(calc_ke_elec, log_psi)
    if x.ndim == 3:
        ke_fn = jax.vmap(ke_fn)
    actual_ke = ke_fn(x)

    np.testing.assert_allclose(actual_ke, target_ke, rtol=1e-6)


@pytest.mark.parametrize("forward_mode", [True, False])
@pytest.mark.parametrize("x", [make_test_x(), make_batched_x()])
def test_kinetic_energy_full(x, forward_mode):
    # single x
    """Test (nabla^2 f)(x) / f(x) for f(x) = sum_i x_i^2 + 3x_i."""
    f, log_f = make_test_log_f()
    log_psi = lambda r, x: log_f(None, r+x)
    mass = jnp.ones(x.shape[-2]) * 2

    # d^2f/dx_i^2 = 2 for all i, so the Laplacian is simply 2 * n, where n is
    # the number of coordiantes. We then divide by f(x) to get (nabla^2 f) / f
    target_ke = -0.5 * (1 + 1/2) * (x.shape[-1] * x.shape[-2]) * 2 / f(None, 2*x)
    ke_fn = partial(calc_ke_full, log_psi, mass, forward_mode=forward_mode)
    if x.ndim == 3:
        ke_fn = jax.vmap(ke_fn)
    actual_ke = ke_fn(x, x)

    np.testing.assert_allclose(actual_ke, target_ke, rtol=1e-6)


_raw_p = -jnp.arange(1,4)
@pytest.mark.parametrize("forward_mode", [True, False])
@pytest.mark.parametrize("p", [_raw_p, _raw_p*1j, _raw_p*1j + 2])
def test_kinetic_energy_compelx(p, forward_mode):
    # psi = exp[p @ x]
    def log_psi_fn(x):
        exp = jnp.exp(x @ p).reshape(1, 1)
        sign, logd = jnp.linalg.slogdet(exp)
        return logd + (jnp.log(sign) if jnp.iscomplexobj(sign) else 0.)

    key0 = jax.random.PRNGKey(0)
    xx = jax.random.uniform(key0, (1, p.shape[-1]))

    # target ke should be -0.5 * p**2
    target_ke = jnp.sum(-0.5 * p**2)
    actual_ke = calc_ke_elec(log_psi_fn, xx, forward_mode=forward_mode)

    np.testing.assert_allclose(actual_ke, target_ke)


@pytest.mark.parametrize("forward_mode", [True, False])
def test_kinetic_energy_spin(forward_mode):
    # psi = exp[p @ x]
    def log_psi_fn(x):
        x, s = split_spin(x)
        return (x * s[:, None]).sum()

    key0 = jax.random.PRNGKey(0)
    xx = jax.random.uniform(key0, (5, 3))
    ss = jax.random.uniform(key0, (5,))
    x = (xx, ss)

    # target ke should be -0.5 * p**2
    target_ke = jnp.sum(-0.5 * ss**2) * 3
    actual_ke = calc_ke_elec(log_psi_fn, x, forward_mode=forward_mode)

    np.testing.assert_allclose(actual_ke, target_ke)


@pytest.mark.parametrize("x", [make_test_x(), make_batched_x()])
def test_local_energy_shape(x):
    f, log_f = make_test_log_f()
    log_psi = partial(log_f, None)
    nuclei, elems = make_test_ions()

    if x.ndim <= 2:
        le = calc_local_energy(log_psi, elems, nuclei, x)
    else:
        le = jax.vmap(partial(calc_local_energy, log_psi, elems, nuclei))(x)
    assert le.shape == f(None, x).shape
