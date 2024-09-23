import functools
from typing import Optional, Sequence, Tuple, Union

import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp

from ..utils import (Array, ElecConf, _t_real, ensure_no_spin,
                     fix_init, gen_kidx, log_linear_exp)
from .base import ElecWfn
from .neuralnet_pbc import dist_features_pbc


def heg_rs(cell: Array, nelec: int) -> float:
    """Calculate density parameter r_s of the homogeneous electron gas
    """
    ndim = len(cell)
    vol = jnp.abs(jnp.linalg.det(cell))
    vol_pp = vol / nelec
    rs = ((2 * (ndim-1) * jnp.pi) / (ndim * vol_pp)) ** (-1./ndim)
    return rs


class ElecProductModel(ElecWfn):
    submodels: Sequence[nn.Module]
    backflow: Optional[nn.Module] = None
    apply_backflow: Union[bool, Sequence[bool]] = True

    def setup(self) -> None:
        self.bf_mask = ([self.apply_backflow] * len(self.submodels)
                        if isinstance(self.apply_backflow, bool)
                        else self.apply_backflow)

    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        sign, logf = 1., 0.
        x_bf = x
        if self.backflow is not None:
            out = self.backflow(x)
            if isinstance(out, tuple):
                x_bf, (sign, logf) = out
            else:
                x_bf = out
        for model, with_bf in zip(self.submodels, self.bf_mask):
            result = model(x_bf if with_bf else x)
            if isinstance(result, tuple):
                sign *= result[0]
                logf += result[1]
            else:
                logf += result
        return sign, logf

class excitedModel(ElecWfn):
    elec_prod_model: nn.Module

    def __call__(self, x:ElecConf):
        return




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
    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        x = ensure_no_spin(x)
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
            return log_linear_exp(sign, logf, weights, axis=0)


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
            split_idx = np.cumsum(self.spins)[:-1]
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
        # work on spins
        if self.spin_symmetry:
            n_orb = max(self.spins)
            if self.full_det:
                raw_orbs = PWDense(n_orb * len(self.spins))(eikx).reshape(-1)
                orbitals = [raw_orbs[_get_fulldet_spin_symm_index(self.spins)]]
            else:
                raw_orbs = PWDense(n_orb)(eikx) # [n_elec, n_orb]
                orb_secs = jnp.split(raw_orbs, split_idx, axis=0)
                orbitals = [raw_orb[:, :n_si]
                            for n_si, raw_orb in zip(self.spins, orb_secs)]
        else: # full determinant (n_elec x n_elec, have off diagonal term)
            pw_secs = jnp.split(eikx, split_idx, axis=0)
            if self.full_det:
                orbitals = [jnp.concatenate([
                    PWDense(n_elec)(pw_si) for pw_si in pw_secs
                ], axis=0)] # [sum(n_si), n_elec]
            else:
                orbitals = [PWDense(n_si)(pw_si)
                            for n_si, pw_si in zip(self.spins, pw_secs)]
        return orbitals


def _get_fulldet_spin_symm_index(spins):
    # get the index to construct the full determinant for spin symmetry case
    # raw orbitals are [n_elec, n_orb, n_comp] and flattened before indexing
    # for 2 component case give the matrix idx of the right -> | a(up) b(up) |
    # a, b: orbitals;  up, dn: electrons (with spin up/dn) --> | b(dn) a(dn) |
    n_elec, n_orb, n_comp = sum(spins), max(spins), len(spins)
    with jax.ensure_compile_time_eval():
        flat_idx = jnp.arange(n_elec * n_orb * n_comp, dtype=int)
        raw_idx = flat_idx.reshape(n_elec, n_orb, n_comp)
        splited = jnp.split(raw_idx, np.cumsum(spins)[:-1], axis=0)
        rolled = jnp.concatenate(
            [jnp.roll(x, i, axis=-1) for i, x in enumerate(splited)], axis=0)
        columns = [rolled[:, :n_si, i] for i, n_si in enumerate(spins)]
        idx = jnp.concatenate(columns, axis=-1)
    assert idx.shape == (n_elec, n_elec)
    return idx


class PairJastrowCCK(ElecWfn):
    spins: Sequence[int]
    cell: Array
    init_jd: float = 1/2 # in the unit of rs
    init_jd_a: float = 1/4

    @nn.compact
    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        x = ensure_no_spin(x)
        # preparation constants
        with jax.ensure_compile_time_eval():
            n_elec, ndim = x.shape
            blkdiag_mask, offdiag_mask = block_diagonal_masks(self.spins, n_elec, True)
            multi_spin = len(self.spins) > 1 and offdiag_mask.sum() > 0
            latvec = self.cell
            rs = heg_rs(latvec, n_elec)
            inv_jd = 1 / (self.init_jd * rs)
            inv_jd_a = 1 / (self.init_jd_a * rs)
        # minimal image distance with sin wrap
        _, _, dist = dist_features_pbc(x, latvec, frac_dist=False, keepdims=False)
        # Jastrow
        jb = 0
        jd = 1 / jnp.abs(self.param("inv_jd", fix_init, inv_jd, _t_real))
        cusp = 1. / (ndim + 1)
        logpsi = - jastrow_CCK(dist[blkdiag_mask], jb, jd, cusp).sum()
        if multi_spin:
            jd = 1 / jnp.abs(self.param("inv_jd_a", fix_init, inv_jd_a, _t_real))
            cusp = 1. / (ndim - 1)
            logpsi -= jastrow_CCK(dist[offdiag_mask], jb, jd, cusp).sum()
        return 1, logpsi


def jastrow_CCK(r, jb, jd, cusp):
    #  (Ceperley, Chester, Kalos 1977)
    ja = 2 * jd**2 * cusp / (1 + 2 * jb**2)
    return ja * jnp.exp(-jb * r) * (1 - jnp.exp(-r / jd)) / r


def block_diagonal_masks(spins: Sequence[int], n_elec: int, triu: bool) -> Tuple[Array]:
    assert sum(spins) == n_elec
    cum_idx = np.cumsum([0, *spins])
    offdiag_mask = jnp.ones((n_elec, n_elec), dtype=bool)
    for i, j in zip(cum_idx[:-1], cum_idx[1:]):
        offdiag_mask = offdiag_mask.at[i:j, i:j].set(False)
    if triu:
        blkdiag_mask = jnp.triu(~offdiag_mask, k=1)
        offdiag_mask = jnp.triu(offdiag_mask)
    else:
        blkdiag_mask = ~offdiag_mask
    return blkdiag_mask, offdiag_mask


class BackflowEtaKCM(nn.Module):

    @nn.compact
    def __call__(self, r: float) -> float:
        # parameters
        lamb = self.param('lamb', fix_init, 0.1, _t_real)
        sb = self.param('sb', fix_init, 0.5, _t_real)
        rb = jnp.abs(self.param('rb', fix_init, 0.05, _t_real))
        wb = jnp.abs(self.param('wb', fix_init, 0.0, _t_real))
        y = backflow_eta_KCM(r, lamb, sb, rb, wb)
        return y


def backflow_eta_KCM(x, lam, sb, rb, wb):
    # Y. Kwon, D.M. Ceperley, R.M. Martin, PRB 48, 12037 (1993)
    # eq. (14), see params in Table I
    # x = r/rs
    nume = 1 + sb * x
    deno = rb + wb * x + x**(7/2)
    return lam * nume / deno


def polynomial_envelope(x: Array, p: int = 6, x_max: float=1.0) -> Array:
    #  smoothly cut off function defined from x \in [0, 1)
    #  arXiv: 2003.03123 eq. (8)
    p1 = p + 1
    p2 = p + 2
    a = - p1 * p2 / 2
    b = p * p2
    c = - p * p1 / 2
    d = x / x_max
    y = 1 + a * d**p + b * d**p1 + c * d**p2
    return jnp.where(d > 1, 0.0, y)


# from e3nn-jax with Apache 2.0
def bessel(x: Array, n: int, x_max: float = 1.0) -> Array:
    r"""Bessel basis functions.

    They obey the following normalization:

    .. math::

        \int_0^c r^2 B_n(r, c) B_m(r, c) dr = \delta_{nm}

    Args:
        x (jax.Array): input of shape ``[...]``
        n (int): number of basis functions
        x_max (float): maximum value of the input

    Returns:
        jax.Array: basis functions of shape ``[..., n]``

    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs;
    ICLR 2020. Equation (7)
    """
    x = jnp.asarray(x)
    assert isinstance(n, int)

    x = x[..., None]
    n = jnp.arange(1, n + 1, dtype=x.dtype)
    x_nonzero = jnp.where(x == 0.0, 1.0, x)
    return jnp.sqrt(2.0 / x_max) * jnp.where(
        x == 0,
        n * jnp.pi / x_max,
        jnp.sin(n * jnp.pi / x_max * x_nonzero) / x_nonzero,
    )


class BackflowEtaBessel(nn.Module):
    nbasis: int
    rcut: float
    npoly_smooth: Optional[int] = 6
    init_scale: Optional[float] = 0.1

    @nn.compact
    def __call__(self, r: float) -> float:
        with jax.ensure_compile_time_eval():
            nbasis = self.nbasis
            rcut = self.rcut
            envelope = polynomial_envelope(r, self.npoly_smooth, rcut)
        coeffs = self.param('coeffs', nn.initializers.normal(self.init_scale), (nbasis,))
        basis = bessel(r, nbasis, rcut)
        return jnp.expand_dims(envelope, -1) * basis@coeffs


class PairBackflow(nn.Module):
    spins: Sequence[int]
    cell: Array
    backflow_eta: nn.Module
    backflow_eta_a: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, x: ElecConf) -> ElecConf:
        x = ensure_no_spin(x)
        n = x.shape[-2]
        with jax.ensure_compile_time_eval():
            latvec = self.cell
            rs = heg_rs(latvec, n)
            spins = self.spins
            blkdiag_mask, offdiag_mask = block_diagonal_masks(spins, n, False)
            eta_para = self.backflow_eta  # parallel spin
            if len(spins) > 1:  # antiparallel spin, default to same as parallel
              eta_anti = self.backflow_eta_a or eta_para
        # displacements
        _, d_hsin, dist = dist_features_pbc(x, latvec, frac_dist=False, keepdims=False)
        x_by_rs = dist / rs
        # !!!! inefficient construction of eta matrix
        eta = eta_para(x_by_rs) * blkdiag_mask * (1-jnp.eye(n))
        if len(self.spins) > 1:
            eta = eta + eta_anti(x_by_rs) * offdiag_mask
        dr = jnp.einsum('ij,ijl->jl', eta, -d_hsin)
        return x + dr


class IterativeBackflow(nn.Module):
    spins: Sequence[int]
    cell: Array
    backflow_etas: list[nn.Module]
    backflow_etas_a: Optional[list[nn.Module]] = None

    @nn.compact
    def __call__(self, x: ElecConf) -> ElecConf:
        x = ensure_no_spin(x)
        with jax.ensure_compile_time_eval():
            spins = self.spins
            eta_paras = self.backflow_etas  # parallel spin
            if len(spins) > 1:  # antiparallel spin, default to same as parallel
                eta_antis = self.backflow_etas_a or eta_paras
            else:
                eta_antis = [None] * len(eta_paras)
        q = x
        for ep, ea in zip(eta_paras, eta_antis):
            q = PairBackflow(
                spins=spins,
                cell=self.cell,
                backflow_eta=ep,
                backflow_eta_a=ea,
            )(q)
        return q
