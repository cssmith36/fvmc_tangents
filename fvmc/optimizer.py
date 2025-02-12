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
# part of this file is borrowed from kfac-jax library
# https://github.com/deepmind/kfac-jax/blob/main/examples/optimizers.py

from functools import partial
from typing import (Any, Callable, Iterator, Mapping, NamedTuple, Optional,
                    Tuple, Union)

import jax
import kfac_jax
import optax
from jax import numpy as jnp
from jax import tree_util as jtu

from . import curvature_tags_and_blocks
from .preconditioner import scale_by_fisher_inverse
from .utils import PMAP_AXIS_NAME

OptaxState = optax.OptState


KFAC_DEFAULTS = dict(
    damping=1e-3,
    momentum=0.0,
    l2_reg=0.0,
    estimation_mode="fisher_exact",
    inverse_update_period=1,
    num_burnin_steps=0,
    register_only_generic=False,
    norm_constraint=1e-3,
    min_damping=1e-4,
    curvature_ema=0.95,
)

SR_DEFAULTS = dict(
    mode="chol",
    damping=1e-3,
    shifting=None,
    max_norm=3e-2,
    proximal=None,
    descender=None,
    momentum=None,
    use_weighted=False,
)


# def counter_transform() -> optax.GradientTransformation:
#     """increase counter and do nothing to gradients"""

#     def init_fn(params):
#         del params
#         return optax.ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

#     def update_fn(updates, state, params=None):
#         del params
#         return updates, optax.ScaleByScheduleState(
#             count=optax.safe_int32_increment(state.count))

#     return optax.GradientTransformation(init_fn, update_fn)


class OptaxWrapper:
    """Wrapper class for Optax optimizers to have the same interface as KFAC.
    """
    def __init__(
            self,
            value_and_grad_func: kfac_jax.optimizer.ValueAndGradFunc,
            optax_optimizer: Callable[..., optax.GradientTransformation],
            value_func_has_aux: bool = False,
            value_func_has_state: bool = False,
            value_func_has_rng: bool = False,
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
          optax_optimizer: The optax optimizer to be wrapped.
          value_func_has_aux: Boolean. Specifies whether the provided callable
            `value_and_grad_func` returns the loss value only, or also some
            auxiliary data. (Default: `False`)
          value_func_has_state: Boolean. Specifies whether the provided callable
            `value_and_grad_func` has a persistent state that is inputted and it
            also outputs an update version of it. (Default: `False`)
          value_func_has_rng: Boolean. Specifies whether the provided callable
            `value_and_grad_func` additionally takes as input an rng key. (Default:
            `False`)
          batch_process_func: Callable. A function which to be called on each batch
            before feeding to the KFAC on device. This could be useful for specific
            device input optimizations. (Default: `None`)
          **optax_kwargs: additional parameters that will be passed to `optax_factory`.
        """
        self._value_and_grad_func = value_and_grad_func
        self._value_func_has_aux = value_func_has_aux
        self._value_func_has_state = value_func_has_state
        self._value_func_has_rng = value_func_has_rng
        self._optax_optimizer = optax_optimizer
        self._batch_process_func = batch_process_func or (lambda x: x)
        self.multi_device = multi_device
        self._pmap_axis_name = pmap_axis_name
        if self.multi_device:
            self._jit_step = jax.pmap(self._step, axis_name=self._pmap_axis_name,
                donate_argnums=list(range(5)))
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
        if self.multi_device:
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
        if self.multi_device:
            stats, grads = jax.lax.pmean((stats, grads), axis_name=self._pmap_axis_name)
        # Compute and apply updates via our optimizer.
        updates, new_state = self._optax_optimizer.update(grads, state, params, data=batch)
        new_params = optax.apply_updates(params, updates)

        # Add step and batch size info
        stats["step"] = getattr(new_state, "count", -1)
        stats["learning_rate"] = getattr(new_state, "hyperparams", {}).get("learning_rate", -1)
        batch_size = jtu.tree_leaves(batch)[0].shape[0]
        stats["batch_size"] = batch_size * jax.device_count()
        stats["data_seen"] = stats["batch_size"] * stats["step"]

        if self._value_func_has_state:
            return new_params, new_state, new_func_state, stats
        else:
            return new_params, new_state, stats

    def step(
            self,
            params: kfac_jax.utils.Params,
            state: OptaxState,
            rng: jnp.ndarray,
            data_iterator: Iterator[kfac_jax.utils.Batch] = None,
            batch: kfac_jax.utils.Batch = None,
            func_state: Optional[kfac_jax.utils.FuncState] = None,
    ) -> Union[Tuple[kfac_jax.utils.Params, Any, kfac_jax.utils.FuncState,
                     Mapping[str, jnp.ndarray]],
               Tuple[kfac_jax.utils.Params, Any,
                     Mapping[str, jnp.ndarray]]]:
        """A step with similar interface to KFAC."""
        if (data_iterator is None) == (batch is None):
            raise ValueError("Exactly one of the arguments ``data_iterator`` and "
                       "``batch`` must be provided.")
        if data_iterator is not None:
            batch = next(data_iterator)
        result = self._jit_step(
            params=params,
            state=state,
            rng=rng,
            batch=batch,
            func_state=func_state,
        )
        return result


def build_lr_schedule(base=1e-4, decay_time=1e4, decay_power=1., warmup_steps=0):
    def decay_schedule(t):
        if decay_power is None or decay_time is None:
            return base
        return base * jnp.power((1.0 / (1.0 + (t/decay_time))), decay_power)
    if not warmup_steps:
        return decay_schedule
    warmup_schedule = optax.linear_schedule(0.1 * base, base, warmup_steps)
    return optax.join_schedules([warmup_schedule, decay_schedule], [warmup_steps])


def build_optimizer(
        value_and_grad_func,
        name,
        lr_schedule=None,
        value_func_has_aux=False,
        value_func_has_state=False,
        value_func_has_rng=False,
        multi_device=False,
        pmap_axis_name=PMAP_AXIS_NAME,
        log_psi_func=None,
        grad_clipping=None,
        **kwargs
):
    # build lr schedule
    if lr_schedule is None:
        lr_schedule = {}
    if isinstance(lr_schedule, float):
        lr_schedule = {"base": lr_schedule}
    if not callable(lr_schedule):
        lr_schedule = build_lr_schedule(**lr_schedule)

    if name == "kfac":
        options = {**KFAC_DEFAULTS, **kwargs}
        const_schedule = {"decay_power": None, "decay_time": None}
        momentum_schedule = build_lr_schedule(**{
            "base": options.pop("momentum", 0.0),
            **const_schedule,
            **options.pop("momentum_schedule", {})
        })
        damping_schedule = build_lr_schedule(**{
            "base": options.pop("damping", 1e-3),
            **const_schedule,
            **options.pop("damping_schedule", {})
        })
        return kfac_jax.Optimizer(value_and_grad_func,
                                  value_func_has_aux=value_func_has_aux,
                                  value_func_has_state=value_func_has_state,
                                  value_func_has_rng=value_func_has_rng,
                                  multi_device=multi_device,
                                  pmap_axis_name=pmap_axis_name,
                                  learning_rate_schedule=lr_schedule,
                                  momentum_schedule=momentum_schedule,
                                  damping_schedule=damping_schedule,
                                  auto_register_kwargs=dict(
                                        graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS),
                                  **options,
                                  )
    else:
        from optax._src import alias as optax_alias

        using_sr = name.lower() in ("sr", "ngd")
        clip_transform = (optax.adaptive_grad_clip(grad_clipping)
                          if grad_clipping else optax.identity())
        if using_sr:
            assert log_psi_func is not None
            options = {**SR_DEFAULTS, **kwargs}
            dname = options.pop("descender")
            momentum = options.pop("momentum")
            shifting = options.pop("shifting")
            max_norm = options.pop("max_norm")
            if shifting is None:
                shifting = lambda t: 1. - lr_schedule(t) / lr_schedule(0)
            norm_schedule = lambda t: max_norm / lr_schedule(t)
            precond = scale_by_fisher_inverse(
                log_psi_fn=log_psi_func,
                shifting=shifting,
                max_norm=norm_schedule,
                pmap_axis_name=pmap_axis_name,
                **options)
            descender = (getattr(optax_alias, dname) if dname
                         else partial(optax.sgd, momentum=momentum))
            opt_factory = lambda learning_rate: optax.chain(
                precond,
                clip_transform,
                descender(learning_rate))
        else:
            opt_factory = lambda learning_rate: optax.chain(
                clip_transform,
                getattr(optax_alias, name)(learning_rate, **kwargs))

        optax_optimizer = optax.inject_hyperparams(opt_factory)(lr_schedule)

        return OptaxWrapper(value_and_grad_func,
                            optax_optimizer=optax_optimizer,
                            value_func_has_aux=value_func_has_aux,
                            value_func_has_state=value_func_has_state,
                            value_func_has_rng=value_func_has_rng,
                            multi_device=multi_device,
                            pmap_axis_name=pmap_axis_name)
