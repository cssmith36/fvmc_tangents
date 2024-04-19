# many test functions are borrowed from vmcnet

from functools import partial

import pytest
import jax
import numpy as np
from jax import numpy as jnp
from jax import tree_util as jtu

from fvmc.optimizer import build_optimizer, kfac_jax
from fvmc.utils import adaptive_split, PAXIS, nn


_key0 = jax.random.PRNGKey(0)


def _train(nsteps, optimizer, sampler, params, key, multi_device=False):
    """Train a model with KFAC and return params and loss for all steps."""
    # Distribute
    # n_device = jax.device_count() if multi_device else 1
    if multi_device:
        params = kfac_jax.utils.replicate_all_local_devices(params)
        key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
        sampler = jax.pmap(sampler)
    rng_split = partial(adaptive_split, multi_device=multi_device)

    key, subkey1, subkey2 = rng_split(key, 3)

    init_data = sampler(subkey1)
    optimizer_state = optimizer.init(params, subkey2, init_data)

    loss_list = []
    for _ in range(nsteps):
        key, subkey1, subkey2 = rng_split(key, 3)
        data = sampler(subkey1)
        params, optimizer_state, stats = optimizer.step(
            params=params,
            state=optimizer_state,
            rng=subkey2,
            batch=data,
        )
        loss = stats["loss"][0] if multi_device else stats["loss"]
        # print(params)
        loss_list.append(loss.item())

    if multi_device:
        params = jtu.tree_map(lambda x: x[0], params)

    return loss_list, params


@pytest.mark.parametrize("name", ["adam"]) # no kfac as it seems not working on linear case
@pytest.mark.parametrize("multi_device", [False, True])
def test_optimizer_linear(name, multi_device):

    target_weight = jnp.array([[2.0, 3.0, 4.0, 5.0, 6.0]])
    n_feature = target_weight.shape[-1]
    target_bias = jnp.zeros(n_feature)

    def target_fn(x):
        """Target function is linear: f(x) = (2x, 3x, 4x, 5x, 6x)."""
        return jnp.dot(x, target_weight)

    dense = nn.Dense(n_feature)

    def loss_fn(params, x):
        prediction = dense.apply(params, x)
        target = target_fn(x)
        kfac_jax.register_squared_error_loss(prediction, target)
        return PAXIS.all_mean(jnp.square(prediction - target))

    loss_and_grad = jax.value_and_grad(loss_fn)

    optimizer = build_optimizer(
        loss_and_grad,
        name=name,
        lr_schedule=lambda x: 1e-1,
        value_func_has_aux=False,
        value_func_has_rng=False,
        value_func_has_state=False,
        multi_device=multi_device,
        pmap_axis_name=PAXIS.name)

    n_sample = 1000
    def sampler(key):
        return jax.random.uniform(key, (n_sample, 1))

    key, subkey = jax.random.split(_key0)
    params = dense.init(subkey, jnp.zeros(1))

    n_iter = 1000
    loss_list, params = _train(n_iter, optimizer, sampler, params, key, multi_device)

    # print(loss_list)
    # print(params)
    assert loss_list[-1] < 1e-7
    np.testing.assert_allclose(params['params']['kernel'], target_weight, atol=1e-5)
    np.testing.assert_allclose(params['params']['bias'], target_bias, atol=1e-5)
