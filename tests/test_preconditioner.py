# many test functions are borrowed from vmcnet

import pytest
import jax
import optax
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from fvmc.preconditioner import scale_by_fisher_inverse, scale_by_fisher_inverse_direct


def _setup_fisher(centered=True):
    energy_grad = jnp.array([0.5, -0.5, 1.2])
    params = jnp.array([1.0, 2.0, 3.0])
    positions = jnp.array(
        [
            [1.0, -0.1, 0.1],
            [-0.1, 1.0, 0.0],
            [0.1, 0.0, 1.0],
            [0.01, -0.01, 0.0],
            [0.0, 0.0, -0.02],
        ]
    )
    n_sample = len(positions)

    jacobian = positions
    if centered:
        centered_jacobian = jacobian - jnp.mean(jacobian, axis=0)
        centered_JT_J = jnp.matmul(jnp.transpose(centered_jacobian), centered_jacobian)
        fisher = centered_JT_J / n_sample  # technically 0.25 * Fisher
    else:
        fisher = jacobian.T @ jacobian / n_sample

    def log_prob_fn(params, positions):
        return 2*jnp.matmul(positions, params)

    return energy_grad, params, positions, fisher, log_prob_fn


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("damping,mixing", [(0., 0.), (1e-3, 0.9)])
def test_fisher_inverse_cg(damping, mixing, weighted):
    """Check that the fisher inverse fn in lazy mode produces the solution to Fx = b."""
    (
        energy_grad,
        params,
        positions,
        fisher,
        log_prob_fn
    ) = _setup_fisher()

    fisher_precond = scale_by_fisher_inverse(
        log_prob_fn,
        damping=damping,
        mixing_factor=mixing,
        use_weighted=weighted)
    state = fisher_precond.init(params)
    if mixing > 0:
        state = state._replace(last_grads_flat=energy_grad, mixing_factor=mixing)

    data = (positions, log_prob_fn(params, positions)) if weighted else positions
    finv_grad, new_state = fisher_precond.update(energy_grad, state, params, data)

    np.testing.assert_allclose(
        ravel_pytree(finv_grad)[0], new_state[0], rtol=1e-12, atol=0.)

    np.testing.assert_allclose(
        (fisher + damping*jnp.eye(3)) @ finv_grad, energy_grad, atol=1e-6)


@pytest.mark.parametrize("centered", [True, False])
@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
def test_fisher_with_optax(centered, direct, weighted):
    (
        energy_grad,
        params,
        positions,
        fisher,
        log_prob_fn
    ) = _setup_fisher(centered)

    precond = scale_by_fisher_inverse(
        log_prob_fn,
        damping=0,
        centered=centered,
        use_weighted=weighted,
        direct_inverse=direct)

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
