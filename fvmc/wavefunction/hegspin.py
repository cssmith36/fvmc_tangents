import functools
from dataclasses import field as _field
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import numpy as onp
from flax import linen as nn
from jax import numpy as jnp

from ..utils import (Array, ElecConf, _t_real, ensure_spin, split_spin,
                     gen_kidx, log_linear_exp)
from .base import ElecWfn
from .neuralnet_pbc import dist_features_pbc
from .heg import block_diagonal_masks, heg_rs, jastrow_CCK, fix_init


class PlanewaveSlaterSpin(ElecWfn):
    cell: Array
    multi_det: Optional[int] = None
    n_k: Optional[int] = None
    twist: Optional[Array] = None
    close_shell: bool = True
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        x = ensure_spin(x)
        kwargs = dict(
            cell=self.cell, n_k=self.n_k, twist=self.twist,
            close_shell=self.close_shell,
            init_scale=self.init_scale)
        # reduce over determinants
        def det_reducer(carry, orbital):
            sign, logf = carry
            news, newl = jnp.linalg.slogdet(orbital)
            return sign * news, logf + newl
        # handle potential multi-determinant
        if not self.multi_det or self.multi_det <= 1:
            orbitals = PlanewaveOrbitalSpin(**kwargs)(x)
            return functools.reduce(det_reducer, orbitals, (1., 0.))
        else:
            orbitals = nn.vmap(
                PlanewaveOrbitalSpin,
                in_axes=None, out_axes=0,
                axis_size=self.multi_det,
                variable_axes={"params": 0},
                split_rngs={"params": True},
            )(**kwargs)(x)
            sign, logf = functools.reduce(det_reducer, orbitals, (1., 0.))
            weights = self.param("det_weights", nn.initializers.ones, (self.multi_det,))
            return log_linear_exp(sign, logf, weights, axis=0)


class PlanewaveOrbitalSpin(nn.Module):
    cell: Array
    n_k: Optional[int] = None
    twist: Optional[Array] = None
    close_shell: bool = True
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, x: ElecConf) -> Sequence[Array]:
        x, s = split_spin(x)
        with jax.ensure_compile_time_eval():
            n_elec, n_dim = x.shape
            raw_n_k = self.n_k or n_elec
            recvec = jnp.linalg.inv(self.cell).T
            # this will take the desired k-points rounded up to a closed shell
            kpts = jnp.asarray(gen_kidx(n_dim, raw_n_k, close_shell=self.close_shell))
            if self.twist is not None:
                twist = (jnp.asarray(self.twist) + 0.5) % 1. - 0.5
                kpts += twist
            kvecs = 2 * jnp.pi * kpts @ recvec # [n_k, n_dim]
        # plane wave
        eikx = jnp.exp(1j * x @ kvecs.T) # [n_elec, n_k]
        # dense for coefficients
        PWDense = functools.partial(
            nn.Dense, use_bias=False, param_dtype=_t_real,
            kernel_init=nn.initializers.normal(self.init_scale))
        # full spinor determinant [n_elec, n_elec]
        n_orb, n_comp = n_elec, 2
        spin_orbs = nn.vmap(
            PWDense,
            in_axes=None, out_axes=-1, axis_size=n_comp,
            variable_axes={"params": -1},
            split_rngs={"params": True}
        )(n_orb)(eikx) # [n_elec, n_orb, n_comp]
        # einsum might be slow but keep here for clarity
        orbitals = [jnp.einsum("ijs,is->ij", spin_orbs, self._sbasis(s))]
        return orbitals

    def _sbasis(self, s):
        # return a spin basis that has shape [n_elec, n_comp]
        # for the current case, use e^{+-is} for 2 components (spin up/down)
        return jnp.exp(1j * jnp.array([1, -1]) * s[:, None])


class PairJastrowCCKSpin(ElecWfn):
    cell: Array
    init_jd: float = 2./3 # in the unit of rs
    optimize_cusp: bool = False

    @nn.compact
    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        x, _ = split_spin(x)
        # preparation constants
        with jax.ensure_compile_time_eval():
            n_elec, ndim = x.shape
            blkdiag_mask, _ = block_diagonal_masks([n_elec], n_elec, True)
            latvec = self.cell
            rs = heg_rs(latvec, n_elec)
        # minimal image distance with sin wrap
        _, _, dist = dist_features_pbc(x, latvec, frac_dist=False, keepdims=False)
        dist /= rs # dist in units of rs
        # Jastrow
        jb = 0
        jd = jnp.abs(self.param("jastrow_d", fix_init, self.init_jd, _t_real))
        cusp = 1 / (ndim - 1) # cusp for anti parallel spins
        if self.optimize_cusp:
            cusp = jnp.abs(self.param("cusp", fix_init, cusp, _t_real))
        cusp *= rs
        logpsi = - jastrow_CCK(dist[blkdiag_mask], jb, jd, cusp).sum()
        return 1, logpsi
