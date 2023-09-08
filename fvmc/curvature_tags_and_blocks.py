# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# this file is modified from ferminet library
# https://github.com/deepmind/ferminet/blob/main/ferminet/curvature_tags_and_blocks.py

"""Curvature blocks for FermiNet."""
from typing import Any, Mapping, Optional, Sequence, Union
import chex
import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np


vmap_psd_inv_cholesky = jax.vmap(kfac_jax.utils.psd_inv_cholesky, (0, None), 0)
vmap_matmul = jax.vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)


repeated_dense_tag = kfac_jax.LayerTag("repeated_dense_tag", 1, 1)


def register_repeated_dense(y, x, w, b):
    if b is None:
        return repeated_dense_tag.bind(y, x, w)
    return repeated_dense_tag.bind(y, x, w, b)


class RepeatedDenseBlock(kfac_jax.DenseTwoKroneckerFactored):
  """Dense block that is repeatedly applied to multiple inputs (e.g. vmap)."""

  def fixed_scale(self) -> chex.Numeric:
    (x_shape,) = self.inputs_shapes
    return float(kfac_jax.utils.product(x_shape) // (x_shape[0] * x_shape[-1]))

  def update_curvature_matrix_estimate(
      self,
      state: kfac_jax.TwoKroneckerFactored.State,
      estimation_data: Mapping[str, Sequence[chex.Array]],
      ema_old: chex.Numeric,
      ema_new: chex.Numeric,
      batch_size: int,
      *args,
      **kwargs
  ) -> kfac_jax.TwoKroneckerFactored.State:
    estimation_data = dict(**estimation_data)
    x, = estimation_data["inputs"]
    dy, = estimation_data["outputs_tangent"]
    assert x.shape[0] == batch_size
    estimation_data["inputs"] = (x.reshape([-1, x.shape[-1]]),)
    estimation_data["outputs_tangent"] = (dy.reshape([-1, dy.shape[-1]]),)
    batch_size = x.size // x.shape[-1]
    return super().update_curvature_matrix_estimate(
        state,
        estimation_data,
        ema_old,
        ema_new,
        batch_size,
        *args,
        **kwargs
    )
            

def _dense(x: chex.Array, params: Sequence[chex.Array]) -> chex.Array:
    """Example of a dense layer function."""
    w, *opt_b = params
    y = jnp.matmul(x, w)
    return y if not opt_b else y + opt_b[0]


def _dense_with_reshape(x: chex.Array, params: Sequence[chex.Array]) -> chex.Array:
    w, b = params
    y = jnp.matmul(x, w)
    return y + b.reshape((1,) * (y.ndim - 1) + (-1,))


def _dense_parameter_extractor(
        eqns: Sequence[jax.core.JaxprEqn],
) -> Mapping[str, Any]:
    """Extracts all parameters from the conv_general_dilated operator."""
    for eqn in eqns:
        if eqn.primitive.name == "dot_general":
            return dict(**eqn.params)
    assert False


def _make_repeat_dense_pattern(
        batch_dims: int,
        with_bias: bool,
        reshape: bool,
        in_dim: int = 13,
        out_dim: int = 5,
        extra_dims: Sequence[int] = tuple(range(6, 13))
) -> kfac_jax.tag_graph_matcher.GraphPattern:
    example_fn = _dense_with_reshape if with_bias and reshape else _dense
    for i in range(batch_dims):
        example_fn = jax.vmap(example_fn, 
            in_axes=[0, ([None, None] if with_bias else [None])])
    x_shape = [*extra_dims[:batch_dims+1], in_dim]
    p_shapes = ([[in_dim, out_dim], [out_dim]] if with_bias else
                [[in_dim, out_dim]])
    return kfac_jax.tag_graph_matcher.GraphPattern(
        name=f"repeated_dense{batch_dims}_"
             f"{'with' if with_bias else 'no'}_bias"
             f"{'_reshape' if reshape else ''}",
        tag_primitive=repeated_dense_tag,
        compute_func=example_fn,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[np.zeros(x_shape), [np.zeros(s) for s in p_shapes]],
    )
    

GRAPH_PATTERNS = tuple(
    _make_repeat_dense_pattern(n, b, s)
    for n in range(1, 6)
    for b, s in ((True, True), (True, False), (False, False))
) + kfac_jax.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS


kfac_jax.set_default_tag_to_block_ctor("repeated_dense_tag", RepeatedDenseBlock)