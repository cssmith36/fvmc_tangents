# many test functions are borrowed from vmcnet

import pytest
import jax
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from fvmc.preconditioner import build_fisher_preconditioner


def _setup_fisher():
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
    centered_jacobian = jacobian - jnp.mean(jacobian, axis=0)
    centered_JT_J = jnp.matmul(jnp.transpose(centered_jacobian), centered_jacobian)
    fisher = centered_JT_J / n_sample  # technically 0.25 * Fisher

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

    fisher_precond = build_fisher_preconditioner(
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
