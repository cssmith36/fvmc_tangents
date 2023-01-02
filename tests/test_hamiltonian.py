# many test functions are borrowed from vmcnet

import pytest
import numpy as np
from jax import numpy as jnp
from functools import partial

from vdmc.hamiltonian import calc_potential_energy, calc_kinetic_energy, calc_local_energy


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
    ions, charges = make_test_ions()

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
    actual_pe = calc_potential_energy(ions, charges, x)

    np.testing.assert_allclose(actual_pe, target_pe)


@pytest.mark.parametrize("x", [make_test_x(), make_batched_x()])
def test_kinetic_energy(x):
    # single x
    """Test (nabla^2 f)(x) / f(x) for f(x) = sum_i x_i^2 + 3x_i."""
    f, log_f = make_test_log_f()
    log_psi = partial(log_f, None)

    # d^2f/dx_i^2 = 2 for all i, so the Laplacian is simply 2 * n, where n is
    # the number of coordiantes. We then divide by f(x) to get (nabla^2 f) / f
    target_ke = -0.5 * (x.shape[-1] * x.shape[-2]) * 2 / f(None, x)
    actual_ke = calc_kinetic_energy(log_psi, x)

    np.testing.assert_allclose(actual_ke, target_ke, rtol=1e-6)


@pytest.mark.parametrize("x", [make_test_x(), make_batched_x()])
def test_local_energy_shape(x):
    f, log_f = make_test_log_f()
    log_psi = partial(log_f, None)
    ions, charges = make_test_ions()

    le = calc_local_energy(log_psi, ions, charges, x)
    assert le.shape == f(None, x).shape
    