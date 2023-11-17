import functools
from dataclasses import field as _field
from typing import Optional, Sequence, Tuple

import jax
import numpy as onp
from flax import linen as nn
from jax import numpy as jnp

from ..utils import Array, _t_real, displace_matrix, fix_init, gen_kidx
from .base import ElecWfn


class ElecProductModel(ElecWfn):
    submodels: Sequence[nn.Module]
    backflow: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        if self.backflow is not None:
            x = self.backflow(x)
        sign, logf = 1., 0.
        for model in self.submodels:
            result = model(x)
            if isinstance(result, tuple):
                sign *= result[0]
                logf += result[1]
            else:
                logf += result
        return sign, logf


class PlanewaveSlater(ElecWfn):
    spins: Sequence[int]
    cell: Array
    multi_det: Optional[int] = None
    n_k: Optional[int] = None
    twist: Optional[Array] = None
    close_shell: bool = True
    spin_symmetry: bool = True
    full_det: bool = False
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        kwargs = dict(
            spins=self.spins, cell=self.cell, n_k=self.n_k, twist=self.twist,
            close_shell=self.close_shell, spin_symmetry=self.spin_symmetry,
            full_det=self.full_det, init_scale=self.init_scale)
        # reduce over determinants
        def det_reducer(carry, orbital):
            sign, logf = carry
            news, newl = jnp.linalg.slogdet(orbital)
            return sign * news, logf + newl
        # handle potential multi-determinant
        if not self.multi_det or self.multi_det <= 1:
            orbitals = PlanewaveOrbital(**kwargs)(x)
            return functools.reduce(det_reducer, orbitals, (1., 0.))
        else:
            orbitals = nn.vmap(
                PlanewaveOrbital,
                in_axes=None, out_axes=0,
                axis_size=self.multi_det,
                variable_axes={"params": 0},
                split_rngs={"params": True},
            )(**kwargs)(x)
            sign, logf = functools.reduce(det_reducer, orbitals, (1., 0.))
            weights = self.param("det_weights", nn.initializers.ones, (self.multi_det,))
            return jax.nn.logsumexp(logf, b=weights*sign, return_sign=True)[::-1]


class PlanewaveOrbital(nn.Module):
    spins: Sequence[int]
    cell: Array
    n_k: Optional[int] = None
    twist: Optional[Array] = None
    close_shell: bool = True
    spin_symmetry: bool = True
    full_det: bool = False
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, x: Array) -> Sequence[Array]:
        # preparation constants
        with jax.ensure_compile_time_eval():
            n_elec, n_dim = x.shape
            assert sum(self.spins) == n_elec
            split_idx = onp.cumsum(self.spins)[:-1]
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
        pwDense = functools.partial(
            nn.Dense, use_bias=False, param_dtype=_t_real,
            kernel_init=nn.initializers.normal(self.init_scale))
        # work on spins
        if self.spin_symmetry:
            n_orb = max(self.spins) if not self.full_det else n_elec
            raw_orbs = pwDense(n_orb)(eikx) # [n_elec, n_orb]
            if self.full_det:
                orbitals = [raw_orbs]
            else:
                orb_secs = jnp.split(raw_orbs, split_idx, axis=0)
                orbitals = [raw_orb[:, :n_si]
                            for n_si, raw_orb in zip(self.spins, orb_secs)]
        else: # full determinant (n_elec x n_elec, have off diagonal term)
            pw_secs = jnp.split(eikx, split_idx, axis=0)
            if self.full_det:
                orbitals = [jnp.concatenate([
                    pwDense(n_elec)(pw_si) for pw_si in pw_secs
                ], axis=0)] # [sum(n_si), n_elec]
            else:
                orbitals = [pwDense(n_si)(pw_si)
                            for n_si, pw_si in zip(self.spins, pw_secs)]
        return orbitals


class PairJastrowCCK(ElecWfn):
    spins: Sequence[int]
    cell: Array
    init_jd: float = 1./0.045

    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        # preparation constants
        with jax.ensure_compile_time_eval():
            n_elec, ndim = x.shape
            assert sum(self.spins) == n_elec
            cum_idx = onp.cumsum([0, *self.spins])
            offdiag_mask = jnp.ones((n_elec, n_elec), dtype=bool)
            for i, j in zip(cum_idx[:-1], cum_idx[1:]):
                offdiag_mask = offdiag_mask.at[i:j, i:j].set(False)
            blkdiag_mask = jnp.triu(~offdiag_mask, k=1)
            multi_spin = len(self.spins) > 1 and offdiag_mask.sum() > 0
            # minimal image distance with sin wrap
            latvec = self.cell
            invvec = jnp.linalg.inv(latvec)
        x_frac = x @ invvec
        d_frac = displace_matrix(x_frac, x_frac)
        d_frac = (d_frac + 0.5) % 1. - 0.5
        d_hsin = jnp.sin(jnp.pi * d_frac) @ (latvec / jnp.pi)
        dist = jnp.linalg.norm(d_hsin + jnp.eye(n_elec)[..., None], axis=-1)
        # Jastrow
        jb = 0
        jd = 1 / self.param("inv_jd", fix_init, 1/self.init_jd, _t_real)
        cusp = 1. / (ndim + 1)
        logpsi = -jastrow_CCK(dist[blkdiag_mask], jb, jd, cusp).sum()
        if multi_spin:
            jd = 1 / self.param("inv_jd_a", fix_init, 2/self.init_jd, _t_real)
            cusp = 1. / (ndim - 1)
            logpsi -= jastrow_CCK(dist[offdiag_mask], jb, jd, cusp).sum()
        return 1, logpsi

def jastrow_CCK(r, jb, jd, cusp):
    #  (Ceperley, Chester, Kalos 1977)
    A = 2 * jd**2 * cusp / (1 + 2 * jb**2)
    return A * jnp.exp(-jb * r) * (1 - jnp.exp(-r / jd)) / r
