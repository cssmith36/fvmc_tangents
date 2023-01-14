# Copyright DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# this file is borrowed from kfac-jax library
# https://github.com/deepmind/kfac-jax/blob/main/examples/optimizers.py

import kfac_jax
import optax
from typing import Optional, Any, Callable, Tuple, Mapping, Union
import jax
import jax.numpy as jnp
from . import curvature_tags_and_blocks
from .utils import paxis


OptaxState = Any


KFAC_DEFAULTS = dict(
    l2_reg=0.0,
    estimation_mode="fisher_exact",
    inverse_update_period=1,
    num_burnin_steps=0,
    register_only_generic=False,
    norm_constraint=1e-3,
    min_damping=1e-4,
    curvature_ema=0.95,
)

class OptaxWrapper:
    """Wrapper class for Optax optimizers to have the same interface as KFAC.
    """
    def __init__(
            self,
            value_and_grad_func: kfac_jax.optimizer.ValueAndGradFunc,
            value_func_has_aux: bool,
            value_func_has_state: bool,
            value_func_has_rng: bool,
            optax_optimizer: optax.GradientTransformation,
            multi_device: bool = False,
            pmap_axis_name: str = "optax_axis",
            batch_process_func: Optional[Callable[[Any], Any]] = None,
    ):
        """Initializes the Optax wrapper.

        Args:
          value_and_grad_func: Python callable. The function should return the value
            of the loss to be optimized and its gradients. If the argument
            `value_func_has_aux` is `False` then the interface should be:
              loss, loss_grads = value_and_grad_func(params, batch)
            If `value_func_has_aux` is `True` then the interface should be:
              (loss, aux), loss_grads = value_and_grad_func(params, batch)
          value_func_has_aux: Boolean. Specifies whether the provided callable
            `value_and_grad_func` returns the loss value only, or also some
            auxiliary data. (Default: `False`)
          value_func_has_state: Boolean. Specifies whether the provided callable
            `value_and_grad_func` has a persistent state that is inputted and it
            also outputs an update version of it. (Default: `False`)
          value_func_has_rng: Boolean. Specifies whether the provided callable
            `value_and_grad_func` additionally takes as input an rng key. (Default:
            `False`)
          optax_optimizer: The optax optimizer to be wrapped.
          batch_process_func: Callable. A function which to be called on each batch
            before feeding to the KFAC on device. This could be useful for specific
            device input optimizations. (Default: `None`)
        """
        self._value_and_grad_func = value_and_grad_func
        self._value_func_has_aux = value_func_has_aux
        self._value_func_has_state = value_func_has_state
        self._value_func_has_rng = value_func_has_rng
        self._optax_optimizer = optax_optimizer
        self._batch_process_func = batch_process_func or (lambda x: x)
        self._multi_device = multi_device
        self._pmap_axis_name = pmap_axis_name
        if self._multi_device:
            self._jit_step = jax.pmap(self._step, axis_name=self._pmap_axis_name, donate_argnums=list(range(5)))
        else:
            self._jit_step = jax.jit(self._step)

    def init(
            self,
            params: kfac_jax.utils.Params,
            rng: jnp.ndarray,
            batch: kfac_jax.utils.Batch,
            func_state: Optional[kfac_jax.utils.FuncState] = None
    ) -> OptaxState:
        """Initializes the optimizer and returns the appropriate optimizer state."""
        del rng, batch, func_state
        if self._multi_device:
            return jax.pmap(self._optax_optimizer.init, axis_name=self._pmap_axis_name)(params)
        else:
            return self._optax_optimizer.init(params)

    def _step(
            self,
            params: kfac_jax.utils.Params,
            state: OptaxState,
            rng,
            batch: kfac_jax.utils.Batch,
            func_state: Optional[kfac_jax.utils.FuncState] = None,
    ) -> kfac_jax.optimizer.FuncOutputs:
        """A single step of optax."""
        batch = self._batch_process_func(batch)
        func_args = kfac_jax.optimizer.make_func_args(
            params, func_state, rng, batch,
            has_state=self._value_func_has_state,
            has_rng=self._value_func_has_rng
        )
        out, grads = self._value_and_grad_func(*func_args)

        if not self._value_func_has_aux and not self._value_func_has_state:
            loss, new_func_state, aux = out, None, {}
        else:
            loss, other = out
            if self._value_func_has_aux and self._value_func_has_state:
                new_func_state, aux = other
            elif self._value_func_has_aux:
                new_func_state, aux = None, other
            else:
                new_func_state, aux = other, {}
        stats = dict(loss=loss, aux=aux)
        if self._multi_device:
            stats, grads = jax.lax.pmean((stats, grads), axis_name=self._pmap_axis_name)
        # Compute and apply updates via our optimizer.
        updates, new_state = self._optax_optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)

        # Add batch size
        batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
        stats["batch_size"] = batch_size * jax.device_count()

        if self._value_func_has_state:
            return new_params, new_state, new_func_state, stats
        else:
            return new_params, new_state, stats

    def step(
            self,
            params: kfac_jax.utils.Params,
            state: OptaxState,
            rng: jnp.ndarray,
            batch: kfac_jax.utils.Batch,
            func_state: Optional[kfac_jax.utils.FuncState] = None,
    ) -> Union[Tuple[kfac_jax.utils.Params, Any, kfac_jax.utils.FuncState,
                     Mapping[str, jnp.ndarray]],
               Tuple[kfac_jax.utils.Params, Any,
                     Mapping[str, jnp.ndarray]]]:
        """A step with similar interface to KFAC."""
        result = self._jit_step(
            params=params,
            state=state,
            rng=rng,
            batch=batch,
            func_state=func_state,
        )
        return result


def make_lr_schedule(base=1e-4, decay_time=1e4, decay_power=1.):
    if decay_power is None or decay_time is None:
        return lambda t: base
    return lambda t: base * jnp.power((1.0 / (1.0 + (t/decay_time))), decay_power)


def make_optimizer(value_and_grad_func,
                   name,
                   lr_schedule=None,
                   value_func_has_aux=False,
                   value_func_has_state=False,
                   value_func_has_rng=False,
                   multi_device=False,
                   **kwargs):
    # build lr schedule
    if lr_schedule is None:
        lr_schedule = {}
    if not callable(lr_schedule):
        lr_schedule = make_lr_schedule(**lr_schedule)

    if name == "kfac":
        const_schedule = {"decay_power": None, "decay_time": None}
        momentum_schedule = make_lr_schedule(**{
            "base": kwargs.pop("momentum", 0.0),
            **const_schedule,
            **kwargs.pop("momentum_schedule", {})
        })
        damping_schedule = make_lr_schedule(**{
            "base": kwargs.pop("damping", 1e-3),
            **const_schedule,
            **kwargs.pop("damping_schedule", {})
        })
        options = {**KFAC_DEFAULTS, **kwargs}
        return kfac_jax.Optimizer(value_and_grad_func,
                                  value_func_has_aux=value_func_has_aux,
                                  value_func_has_state=value_func_has_state,
                                  value_func_has_rng=value_func_has_rng,
                                  multi_device=multi_device,
                                  pmap_axis_name=paxis.name,
                                  learning_rate_schedule=lr_schedule,
                                  momentum_schedule = momentum_schedule,
                                  damping_schedule = damping_schedule,
                                  auto_register_kwargs=dict(
                                        graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS),
                                  **options,
                                  )
    else:
        from optax._src import alias as optax_alias
        opt_fn = getattr(optax_alias, name)
        optax_optimizer = opt_fn(lr_schedule, **kwargs)
        return OptaxWrapper(value_and_grad_func,
                            value_func_has_aux=value_func_has_aux,
                            value_func_has_state=value_func_has_state,
                            value_func_has_rng=value_func_has_rng,
                            multi_device=multi_device,
                            pmap_axis_name=paxis.name,
                            optax_optimizer=optax_optimizer)
