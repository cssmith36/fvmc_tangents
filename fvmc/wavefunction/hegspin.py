import functools
from typing import Optional, Sequence, Tuple

import jax
from flax import linen as nn
from jax import numpy as jnp

from ..utils import (Array, ElecConf, _t_real, ensure_spin, split_spin,
                     gen_kidx, log_linear_exp)
from .base import ElecWfn
from .neuralnet_pbc import dist_features_pbc
from .heg import (ElecProductModel, block_diagonal_masks,
                  heg_rs, jastrow_CCK, fix_init)


class ElecProductModelSpin(ElecProductModel):
    """Product wavefunction that pauli matrices can act on"""

    def pauli(self, x: ElecConf) -> Array:
        """apply Pauli matrices (for each electron) to the wavefunction

        Return an Array of shape (n_elec, 3) corresponding to
        sigma^i_{x|y|z} psi(r,s) / psi(r,s) for each electron i.
        To use this method, one and only one of the submodels should have
        the method `pauli` implemented.
        """
        spin_model = self.get_spin_submodel()
        sm_idx = self.submodels.index(spin_model)
        if self.backflow is not None and self.bf_mask[sm_idx]:
            x = self.backflow(x)
            if isinstance(x, tuple):
                x = x[0]
        return spin_model.pauli(x)

    def slog_and_pauli(self, x: ElecConf) -> Tuple[Tuple[Array, Array], Array]:
        """return the sign and log of the wavefunction, and the Pauli matrices

        See `pauli` method for details. This method ensures the backflow is
        called only once (although jax complier should have figured it out).
        """
        sign, logf = 1., 0.
        # backflow
        x_bf = x
        if self.backflow is not None:
            out = self.backflow(x)
            if isinstance(out, tuple):
                x_bf, (sign, logf) = out
            else:
                x_bf = out
        # iter over models
        spin_model = self.get_spin_submodel() # one and only one
        for model, with_bf in zip(self.submodels, self.bf_mask):
            x_m = x_bf if with_bf else x
            if model is spin_model: # calculate Pauli with sign and logf
                if hasattr(model, "slog_and_pauli"): # potential fast path
                    result, pauli = model.slog_and_pauli(x_m)
                else: # slow path, but make sure pauli is calculated
                    result = model(x_m)
                    pauli = model.pauli(x_m)
            else: # normal model without pauli
                result = model(x_m)
            if isinstance(result, tuple):
                sign *= result[0]
                logf += result[1]
            else:
                logf += result
        return (sign, logf), pauli

    @nn.nowrap
    def get_spin_submodel(self) -> Optional[nn.Module]:
        """return the index and the submodel that has the method `pauli`"""
        spin_models = [m for m in self.submodels if hasattr(m, "pauli")]
        if len(spin_models) != 1:
            raise ValueError("Exactly one submodel should have the method `pauli`")
        return spin_models[0]


class PlanewaveSlaterSpin(ElecWfn):
    cell: Array
    multi_det: Optional[int] = None
    n_k: Optional[int] = None
    twist: Optional[Array] = None
    close_shell: bool = True
    init_scale: float = 1e-2

    def setup(self) -> None:
        kwargs = dict(
            cell=self.cell, n_k=self.n_k, twist=self.twist,
            close_shell=self.close_shell,
            init_scale=self.init_scale)
        self._is_multi_det = self.multi_det and self.multi_det > 1
        if not self._is_multi_det:
            self.orbital_fn = PlanewaveOrbitalSpin(**kwargs)
        else:
            self.orbital_fn = nn.vmap(
                PlanewaveOrbitalSpin,
                in_axes=None, out_axes=0,
                axis_size=self.multi_det,
                variable_axes={"params": 0},
                split_rngs={"params": True},
            )(**kwargs)
            self.det_weights = self.param(
                "det_weights", nn.initializers.ones, (self.multi_det,))

    @nn.nowrap
    def _sbasis(self, s):
        # return a spin basis that has shape [n_elec, n_comp]
        # for the current case, use e^{+-is} for 2 components (spin up/down)
        return jnp.exp(1j * jnp.array([1, -1]) * s[:, None])

    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        x, s = ensure_spin(x)
        spin_orbs = self.orbital_fn(x) # not a list, since no spin up/dn separation
        return self._slog_from_so(spin_orbs, s)

    def _slog_from_so(self, spin_orbs: Array, s: Array) -> Tuple[Array, Array]:
        # contract spin basis
        orbs = jnp.einsum("...ijs,is->...ij", spin_orbs, self._sbasis(s))
        # calculate determinants
        sign, logf = jnp.linalg.slogdet(orbs)
        # handle potential multi-determinant
        if self._is_multi_det:
            sign, logf = log_linear_exp(sign, logf, self.det_weights, axis=0)
        return sign, logf

    def pauli(self, x: ElecConf) -> Array:
        """apply Pauli matrices (for each electron) to the wavefunction

        Return an Array of shape (n_elec, 3) corresponding to
        sigma^i_{x|y|z} psi(r,s) / psi(r,s) for each electron i,
        with last axis being the components, 0: x, 1: y, 2: z.
        """
        x, s = ensure_spin(x)
        spin_orbs = self.orbital_fn(x)
        return self._pauli_from_so(spin_orbs, s)

    def _pauli_from_so(self, spin_orbs: Array, s: Array) -> Array:
        sb0 = self._sbasis(s)
        sbx = jnp.flip(sb0, axis=-1) # (e^-is, e^is)
        sby = jnp.array([1j, -1j]) * sbx # (ie^-is, -ie^is)
        sbz = jnp.array([1, -1]) * sb0 # (e^is, -e^-is)
        psbas = jnp.stack((sbx, sby, sbz), axis=-1) # (n_elec, 2, 3)
        mat0 = jnp.einsum("...ijs,is->...ij", spin_orbs, sb0)
        pmats = jnp.einsum("...ijs,isp->...ijp", spin_orbs, psbas)
        row_diffs = pmats - mat0[..., None]
        # calculate Pauli matrices
        pauli_fn = (det_row_update if not self._is_multi_det else
                    functools.partial(multi_det_row_update,
                                      det_weights=self.det_weights))
        batch_pauli_fn = jax.vmap(pauli_fn, in_axes=(None, -1), out_axes=-1)
        return batch_pauli_fn(mat0, row_diffs)

    def slog_and_pauli(self, x: ElecConf) -> Tuple[Tuple[Array, Array], Array]:
        x, s = ensure_spin(x)
        spin_orbs = self.orbital_fn(x)
        return self._slog_from_so(spin_orbs, s), self._pauli_from_so(spin_orbs, s)


class PlanewaveOrbitalSpin(nn.Module):
    cell: Array
    n_k: Optional[int] = None
    twist: Optional[Array] = None
    close_shell: bool = True
    init_scale: float = 1e-2

    @nn.compact
    def __call__(self, x: Array) -> Sequence[Array]:
        # x is electron coords, spin is not handled here
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
        return spin_orbs # spin orbital, multi component at last axis


class PairJastrowCCKSpin(ElecWfn):
    cell: Array
    init_jd: float = 1/4 # in the unit of rs
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
            inv_jd = 1 / (self.init_jd * rs)
        # minimal image distance with sin wrap
        _, _, dist = dist_features_pbc(x, latvec, frac_dist=False, keepdims=False)
        # Jastrow
        jb = 0
        jd = 1 / jnp.abs(self.param("inv_jd", fix_init, inv_jd, _t_real))
        cusp = 1 / (ndim - 1) # cusp for anti parallel spins
        if self.optimize_cusp:
            cusp = self.param("cusp", fix_init, cusp, _t_real)
        logpsi = - jastrow_CCK(dist[blkdiag_mask], jb, jd, cusp).sum()
        return 1, logpsi


def det_row_update(mat, row_diffs):
    """batched version for updating a determinant by a row

    Args:
        mat (Array): a square matrix with shape (n, n)
        row_diffs (Array): same shape as mat, each row is the update at that row

    Returns:
        det(A') / det(A), the scale factors from updating those n rows, shape (n,)
    """
    # det(A + u @ v^T) = (1 + v^T @ A^-1 @ u) * det(A)
    # u: one hot on updated row index, v^T: updated row
    # v^T @ A^-1 @ u is the n-th col of v^T @ A^-1 for electron n
    inv = jnp.linalg.inv(mat) # square matrix, "ij" in einsum
    # u is just \delta_jn and is pre-contracted to inv0, ij,jn->in
    updates = 1 + jnp.einsum("ni,in->n", row_diffs, inv)
    return updates


def multi_det_row_update(mat, row_diffs, det_weights):
    """batched version updating sum of multiple determinant by same single row

    Args:
        mat (Array): multiple square matrix with shape (m, n, n)
        row_diffs (Array): same shape as mat, each row is the update at that row

    Returns:
        scale factors from updating sum of determinants by those n rows, shape (n,)
    """
    # multi-determinant, do the same thing for each determinant
    updates = jax.vmap(det_row_update, in_axes=0, out_axes=0)(mat, row_diffs)
    msign, mlogf = jnp.linalg.slogdet(mat) # both shape (m,)
    sign0, logf0 = log_linear_exp(msign, mlogf, det_weights, axis=0) # scalar
    uweights = det_weights[:, None] * updates # shape (m, n)
    signu, logfu= log_linear_exp(msign, mlogf, uweights, axis=0) # (n,)
    return signu / sign0 * jnp.exp(logfu - logf0)
