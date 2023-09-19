# many test functions are borrowed from vmcnet

import jax
import numpy as np
import optax
import pytest
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from fvmc.preconditioner import scale_by_fisher_inverse


def _setup_fisher(shifting=1., complex=False):
    energy_grad = jnp.array([0.5, -0.5, 1.2])
    params = jnp.array([1.0, 2.0, 3.0])
    _key0 = jax.random.PRNGKey(0)
    positions = jax.random.normal(_key0, (5, 3))
    if complex:
        positions += 1j * jax.random.normal(_key0+1, (5, 3))
    n_sample = len(positions)

    jacobian = positions
    centering = 1. - jnp.sqrt(1. - shifting)
    jacobian = jacobian - centering * jnp.mean(jacobian, axis=0)
    fisher = (jacobian.T.conj() @ jacobian).real / n_sample

    def log_psi_fn(params, positions):
        return jnp.matmul(positions, params)

    return energy_grad, params, positions, fisher, log_psi_fn


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("damping,mixing", [(0., 0.), (1e-3, 0.9)])
def test_fisher_inverse_cg(damping, mixing, weighted):
    """Check that the fisher inverse fn in lazy mode produces the solution to Fx = b."""
    (
        energy_grad,
        params,
        positions,
        fisher,
        log_psi_fn
    ) = _setup_fisher()

    fisher_precond = scale_by_fisher_inverse(
        log_psi_fn,
        mode="lazy_cg",
        damping=damping,
        x0_mixing=mixing,
        max_norm=1e10,
        use_weighted=weighted)
    state = fisher_precond.init(params)
    if mixing > 0:
        state = state._replace(solver_state=(energy_grad, mixing))

    data = (positions, 2*log_psi_fn(params, positions)) if weighted else positions
    finv_grad, new_state = fisher_precond.update(energy_grad, state, params, data)

    np.testing.assert_allclose(
        ravel_pytree(finv_grad)[0], new_state[1][0], rtol=1e-12, atol=0.)

    np.testing.assert_allclose(
        (fisher + damping*jnp.eye(3)) @ finv_grad, energy_grad, atol=1e-6)


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("shifting", [1, 0.1])
@pytest.mark.parametrize("mode", ["cg", "direct", "qr", "svd"])
def test_fisher_with_optax(shifting, complex, mode, weighted):
    (
        energy_grad,
        params,
        positions,
        fisher,
        log_psi_fn
    ) = _setup_fisher(shifting, complex)

    damping = 1e-3
    fisher = fisher + damping*jnp.eye(3)

    kwargs = {"precondition": True} if mode == "cg" else {}
    precond = scale_by_fisher_inverse(
        log_psi_fn,
        mode=mode,
        damping=damping,
        shifting=shifting,
        max_norm=1e10,
        use_weighted=weighted,
        **kwargs)

    opt = optax.chain(
        precond,
        optax.sgd(0.1, momentum=0.))

    state = opt.init(params)

    finv = jnp.linalg.inv(fisher)
    key = jax.random.PRNGKey(0)

    for _ in range(3):
        key, subkey = jax.random.split(key)
        grads = jax.random.normal(subkey, energy_grad.shape)
        updates, new_state = opt.update(grads, state, params, data=positions)
        np.testing.assert_allclose(updates, -0.1 * finv @ grads)
