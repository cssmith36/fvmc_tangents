from dataclasses import field as _field
from typing import Tuple

import jax
from flax import linen as nn
from jax import numpy as jnp

from ..utils import (Array, ElecConf, NuclConf, _t_real, ensure_no_spin,
                     build_mlp, cdist, displace_matrix, fix_init, pdist)
from .base import FullWfn, ProductModel


# follow the TwoBodyExpDecay class in vmcnet
class SimpleJastrow(nn.Module):
    r"""Isotropic exponential decay two-body Jastrow model.

    The decay is isotropic in the sense that each electron-nuclei and electron-electron
    term is isotropic, i.e. radially symmetric. The computed interactions are:

        \sum_i(-\sum_j Z_j ||elec_i - ion_j|| + \sum_k Q ||elec_i - elec_k||)

    (no exponential because it we are working with log of wavefunctions.)
    Z_j and Q are parameters that are initialized to be nuclei charges and 1.
    """

    elems: Array

    @nn.compact
    def __call__(self, r: NuclConf, x: ElecConf) -> Array:
        # calculate initial scale, so that it returns 0 if all electrons are on nuclei
        x = ensure_no_spin(x)
        cmat = jnp.expand_dims(self.elems, -1) * jnp.expand_dims(self.elems, -2)
        scale = 0.5 * jnp.sum(pdist(r) * cmat)
        # make z and q parameters
        z = self.param("z", fix_init, self.elems, _t_real)
        q = self.param("q", fix_init, 1.0, _t_real)
        # distance matrices
        d_ei = cdist(x, r)
        d_ee = pdist(x)
        # interaction terms
        corr_ei = jnp.sum(d_ei * z, axis=-1)
        corr_ee = jnp.sum(jnp.triu(d_ee) * q, axis=-1)
        return jnp.sum(corr_ee - corr_ei, axis=-1) + scale


class SimpleOrbital(nn.Module):
    r"""Single particle orbital by a simple resnet

    for each electron i, taking [x_i - R_I, |x_i - R_I|, ...] as input
    and output a vector of size n_orb, correspinding to \psi_k(x_i)
    """

    n_orb: int
    n_hidden: int = 3
    activation: str = "gelu"

    @nn.compact
    def __call__(self, r: NuclConf, x: ElecConf) -> Array:
        # n_el = x.shape[-2]
        x = ensure_no_spin(x)
        n_dim = x.shape[-1]
        n_ion = r.shape[-2]
        n_feature = n_ion * (n_dim + 1)
        resnet = build_mlp(
            [n_feature]*self.n_hidden + [self.n_orb],
            residual=True, activation=self.activation, param_dtype=_t_real)
        # build input features
        disp_ei = displace_matrix(x, r) # [..., n_el, n_ion, 3]
        d_ei = jnp.linalg.norm(disp_ei, axis=-1, keepdims=True) # [..., n_el, n_ion, 1]
        feature = jnp.concatenate([disp_ei, d_ei], axis=-1) # [..., n_el, n_ion, 4]
        feature = feature.reshape(*feature.shape[:-2], n_feature) # [..., n_el, (n_ion * 4)]
        # return the result from MLP
        return resnet(feature) # [..., n_el, n_orb]


class SimpleSlater(FullWfn):
    r"""Slater determinant from single particle orbitals

    Separate the electrons into different spins and calculate orbitals for both.
    if full_det is True, use one large determinant. Otherwise use two small ones.
    Return sign and log of determinant when called.
    """

    spins: tuple[int, int] # difference between alpha and beta spin
    full_det: bool = True
    orbital_type: str = "simple"
    orbital_args: dict = _field(default_factory=dict)

    @nn.compact
    def __call__(self, r: NuclConf, x: ElecConf) -> Tuple[Array, Array]:
        x = ensure_no_spin(x)
        n_el = x.shape[-2]
        n_up, n_dn = self.spins
        assert n_up + n_dn == n_el

        if self.orbital_type == "simple":
            OrbCls = SimpleOrbital
        else:
            raise ValueError("unsupported orbital type")
        orb_up_fn = OrbCls((n_el if self.full_det else n_up), **self.orbital_args)
        orb_dn_fn = OrbCls((n_el if self.full_det else n_dn), **self.orbital_args)

        orb_up = orb_up_fn(r, x[..., :n_up, :]) # [..., n_up, n_el|n_up]
        orb_dn = orb_dn_fn(r, x[..., n_up:, :]) # [..., n_up, n_el|n_dn]

        if self.full_det:
            orb_full = jnp.concatenate([orb_up, orb_dn], axis=-2)
            return jnp.linalg.slogdet(orb_full)
        else:
            sign_up, ldet_up = jnp.linalg.slogdet(orb_up)
            sign_dn, ldet_dn = jnp.linalg.slogdet(orb_dn)
            return sign_up * sign_dn, ldet_up + ldet_dn


def build_jastrow_slater(elems, spins, nuclei, dynamic_nuclei=False,
        full_det=True, orbital_type="simple", orbital_args=None):
    orbital_args = orbital_args or {}
    jastrow = SimpleJastrow(elems)
    slater = SimpleSlater(spins, full_det, orbital_type, orbital_args)
    mlist = [jastrow, slater]
    if dynamic_nuclei:
        mlist.append(NucleiGaussianSlater(nuclei, 0.1))
    model = ProductModel(mlist)
    # elec_model = FixNuclei(model, nuclei)
    return model


class NucleiGaussian(FullWfn):
    r"""Gaussian for nuclei wavefunctions, centered on trainable sites

    The log wavefunction is given by - \sum_i (r_i - r0_i)^2 / (2 * sigma_i^2)
    """

    init_r0: Array
    init_sigma: Array

    @nn.compact
    def __call__(self, r: NuclConf, x: ElecConf) -> Tuple[Array, Array]:
        del x
        r0 = self.param("r0", fix_init, self.init_r0, _t_real)
        sigma = self.param("sigma", fix_init,
                           jnp.reshape(self.init_sigma, (-1, 1)), _t_real)
        return 1., -0.5 * jnp.sum(((r - r0) / sigma)**2)


class NucleiGaussianSlater(FullWfn):
    r"""Gaussian for nuclei wavefunctions with Slater determinant exchange

    The wavefunction is given by Det_ij{ exp[-(r_i - r0_j)^2 / (2 * sigma_j^2)] }
    """

    init_r0: Array
    init_sigma: Array

    @nn.compact
    def __call__(self, r: NuclConf, x: ElecConf) -> Tuple[Array, Array]:
        del x
        # r0: [n_nucl, 3], sigma: [n_nucl, 1]
        r0 = self.param("r0", fix_init, self.init_r0, _t_real)
        sigma = self.param("sigma", fix_init,
                           jnp.reshape(self.init_sigma, (-1, 1)), _t_real)
        # exps: [n_nucl, n_nucl]
        exps = jnp.exp(-0.5 * jnp.sum(((r[:, None] - r0) / sigma)**2, -1))
        return jnp.linalg.slogdet(exps)


class NucleiGaussianSlaterPbc(FullWfn):
    r"""Gaussian for nuclei wavefunctions with Slater determinant exchange

    The wavefunction is given by Det_ij{ exp[-(r_i - r0_j)^2 / (2 * sigma_j^2)] }
    """

    cell: Array
    init_r0: Array
    init_sigma: Array

    @nn.compact
    def __call__(self, r: NuclConf, x: ElecConf) -> Tuple[Array, Array]:
        del x
        # r0: [n_nucl, 3], sigma: [n_nucl, 1]
        r0 = self.param("r0", fix_init, self.init_r0, _t_real)
        sigma = self.param("sigma", fix_init,
                           jnp.reshape(self.init_sigma, (-1, 1)), _t_real)
        # pbc displacement as L/\pi * sin(\pi/L * d)
        latvec = self.cell
        invvec = jnp.linalg.inv(latvec)
        disp = displace_matrix(r, r0)
        d_frac = disp @ invvec
        d_hsin = jnp.sin(jnp.pi * d_frac) @ latvec/jnp.pi
        # exps: [n_nucl, n_nucl]
        exps = jnp.exp(-0.5 * jnp.sum((d_hsin / sigma)**2, -1))
        return jnp.linalg.slogdet(exps)
