from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
from flax import linen as nn
from jax import numpy as jnp

from ..utils import (Array, ElecConf, _t_real, build_mlp, parse_activation,
                     attach_spin, split_spin)
from .base import ElecWfn
from .heg import heg_rs
from .neuralnet_pbc import raw_features_pbc


class NeuralBackflow(nn.Module):
    spins: Sequence[int]
    cell: Array
    single_size: int = 32 # size of h_i, before concat with x0_i
    pair_size: int = 26 # size of h_ij, before concat with x0_ij (size 6 for 2D)
    mlp_width: int = 32 # width of the mlp of the message passing layer
    attn_width: int = 32 # width of the message passing layer m_ij
    backflow_scale: float = 0.03 # hardcoded scale of the backflow shift (in rs)
    backflow_layers: int = 3 # number of message passing layers
    jastrow_width: int = 32 # width of the hidden layers in the jastrow mlp
    jastrow_layers: int = 3 # number of layers in the jastrow mlp
    activation: Union[Callable, str] = "gelu"
    kernel_init: Callable = nn.linear.default_kernel_init

    @nn.compact
    def __call__(self, x: ElecConf) -> Tuple[ElecConf, Tuple[float, float]]:
        # set up constants
        x, _s = split_spin(x)
        n_elec, n_dim = x.shape
        assert not self.spins or sum(self.spins) == n_elec
        actv_fn = parse_activation(self.activation)
        MyDense = partial(nn.Dense, param_dtype=_t_real, kernel_init=self.kernel_init)
        # input independent arrays
        with jax.ensure_compile_time_eval():
            invvec = jnp.linalg.inv(self.cell)
            rs = heg_rs(self.cell, n_elec)
            if not self.spins or len(self.spins) == 1: # all same spin, no need to compute s_ij
                s_ij = jnp.zeros((n_elec, n_elec, 1))
            else:
                assert len(self.spins) == 2, "only support 2 type of spins"
                s_i = jnp.concatenate([jnp.ones(self.spins[0]), -jnp.ones(self.spins[1])])
                s_ij = jnp.outer(s_i, s_i)[..., None]
        # make raw pbc feature (dummy r from empty array)
        d_i, d_ij, dist = raw_features_pbc(x[:0], x, self.cell, 1, frac_dist=True)
        x0_i = d_i[..., :0] # dummy x0_i, empty array
        x0_ij = jnp.concatenate([d_ij, s_ij], axis=-1) # [sinx,siny,cosx,cosy,sind,s_ij]
        # Initialize (raw) h_i and h_ij
        h_i = self.param("h_i", nn.initializers.zeros, (self.single_size,))
        h_i = jnp.broadcast_to(h_i, (n_elec, h_i.shape[-1]))
        h_ij = self.param("h_ij", nn.initializers.zeros, (self.pair_size,))
        h_ij = jnp.broadcast_to(h_ij, (n_elec, n_elec, h_ij.shape[-1]))
        # Run the iterative backflow loop
        for _ in range(self.backflow_layers):
            h_i, h_ij = MessagePassingLayer(
                mlp_width=self.mlp_width, attn_width=self.attn_width,
                activation=actv_fn, kernel_init=self.kernel_init
            )(h_i, h_ij, x0_i, x0_ij)
        # the final displacement vector
        bfdense = MyDense(n_dim, use_bias=False)
        x = x + bfdense(h_i) * self.backflow_scale * rs
        # neural jastrow
        x_pbc = jnp.concatenate(
            [jnp.sin(2 * jnp.pi * x @ invvec), jnp.cos(2 * jnp.pi * x @ invvec)],
            axis=-1)
        x_expand = MyDense(self.single_size)(x_pbc)
        j_in = jnp.concatenate([h_i, actv_fn(x_expand)], axis=-1)
        # skip connection in the jastrow network
        jas_sizes = [self.jastrow_width] * self.jastrow_layers + [1]
        jasmlp = build_mlp(jas_sizes,
                           activation=actv_fn, last_bias=False,
                           residual=True, rescale=True,
                           kernel_init=self.kernel_init,
                           param_dtype=_t_real)
        jastrow = jasmlp(j_in).sum()
        # return the final coordinates and the jastrow
        return attach_spin(x, _s), (1., jastrow)


class MessagePassingLayer(nn.Module):
    mlp_width: int
    attn_width: int
    activation: Union[Callable, str] = "gelu"
    kernel_init: Callable = nn.linear.default_kernel_init

    def setup(self):
        self.Dense = partial(
            nn.Dense,
            param_dtype=_t_real,
            kernel_init=self.kernel_init)
        self.actv_fn = parse_activation(self.activation)

    def omega(self, g_ij):
        # Constructs the attention weight matrix
        # Eq (8) in arXiv:2305.07240
        g_ij = nn.LayerNorm()(g_ij)
        # Construct Q, K & W, NOTE here we use mlp_width, not attn_width
        q_ij = self.Dense(self.mlp_width, use_bias=False)(g_ij)
        k_ij = self.Dense(self.mlp_width, use_bias=False)(g_ij)
        # rescale before apply gelu (divide by sqrt(n_elec))
        scale = jnp.sqrt(g_ij.shape[0]) # n_elec
        w_ij = self.actv_fn(
            q_ij.transpose((2,0,1)) @ k_ij.transpose((2,0,1)) / scale
        ).transpose((1,2,0))
        # Reduce from mlp width to attention width
        # NOTE: this is not the same as the original paper, and may be unnecessary
        w_ij = self.Dense(self.attn_width, use_bias=False)(w_ij)
        return w_ij

    def phi(self, g_ij):
        # Constructs the message to be passed from layer to layer
        # appear in arXiv:2305.07240 when define m_ij at the paragraph above Eq (8)
        return nn.Sequential([
            nn.LayerNorm(),
            self.Dense(self.mlp_width),
            self.actv_fn,
            self.Dense(self.attn_width, use_bias=False)
        ])(g_ij)

    def f1b(self, g_i, m_ij, out_size):
        # MLP transforming the message to the single-particle stream h_i
        # Eq (9) in arXiv:2305.07240
        m_sum = jnp.einsum('ijk->ik', m_ij) - jnp.einsum('iik->ik', m_ij)
        h1_in = jnp.concatenate([m_sum, g_i], axis=-1)
        return nn.Sequential([
            nn.LayerNorm(),
            self.Dense(self.mlp_width),
            self.actv_fn,
            self.Dense(out_size, use_bias=False)
        ])(h1_in)

    def f2b(self, g_ij, m_ij, out_size):
        # MLP transforming the message to the particle-particle stream h_ij
        # Eq (10) in arXiv:2305.07240
        h2_in = jnp.concatenate([m_ij, g_ij], axis=-1)
        return nn.Sequential([
            nn.LayerNorm(),
            self.Dense(self.mlp_width),
            self.actv_fn,
            self.Dense(out_size, use_bias=False)
        ])(h2_in)

    @nn.compact
    def __call__(self, h_i, h_ij, x0_i, x0_ij):
        n_elec = h_i.shape[0]
        assert h_ij.shape[:2] == (n_elec, n_elec)
        g_i = jnp.concatenate([h_i, x0_i], axis=-1)
        g_ij = jnp.concatenate([h_ij, x0_ij], axis=-1)
        # make attention matrix
        w_ij = self.omega(g_ij)
        # Update the message
        m_ij = w_ij * self.phi(g_ij)
        # Update the single-particle and particle-particle streams
        h_i = self.f1b(g_i, m_ij, out_size=h_i.shape[-1]) + h_i
        h_ij = self.f2b(g_ij, m_ij, out_size=h_ij.shape[-1]) + h_ij
        # Final Layer Norm before sending into next layer
        return h_i, h_ij
