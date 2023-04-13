from dataclasses import field as _field
from typing import Optional, Sequence, Tuple, Union

import jax
from jax import numpy as jnp
from flax import linen as nn

from .utils import Array, _t_real, build_mlp, cdist, diffmat, fix_init, pdist


def log_prob_from_model(model: nn.Module):
    return lambda p, *args, **kwargs: 2 * model.apply(p, *args, **kwargs)[1]


# follow the TwoBodyExpDecay class in vmcnet
class Jastrow(nn.Module):
    r"""Isotropic exponential decay two-body Jastrow model.
    
    The decay is isotropic in the sense that each electron-nuclei and electron-electron
    term is isotropic, i.e. radially symmetric. The computed interactions are:

        \sum_i(-\sum_j Z_j ||elec_i - ion_j|| + \sum_k Q ||elec_i - elec_k||)

    (no exponential because it we are working with log of wavefunctions.)
    Z_j and Q are parameters that are initialized to be ion charges and 1.
    """

    ions: Array
    elems: Array

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # calculate initial scale, so that it returns 0 if all electrons are on ions
        cmat = jnp.expand_dims(self.elems, -1) * jnp.expand_dims(self.elems, -2)
        scale = 0.5 * jnp.sum(pdist(self.ions) * cmat)
        # make z and q parameters
        z = self.param("z", fix_init, self.elems, _t_real)
        q = self.param("q", fix_init, 1.0, _t_real)
        # distance matrices
        r_ei = cdist(x, self.ions)
        r_ee = pdist(x)
        # interaction terms
        corr_ei = jnp.sum(r_ei * z, axis=-1)
        corr_ee = jnp.sum(jnp.triu(r_ee) * q, axis=-1)
        return jnp.sum(corr_ee - corr_ei, axis=-1) + scale

    
class SimpleOrbital(nn.Module):
    r"""Single particle orbital by a simple resnet

    for each electron i, taking [x_i - R_I, |x_i - R_I|, ...] as input
    and output a vector of size n_orb, correspinding to \psi_k(x_i)
    """

    ions: Array
    n_orb: int
    n_hidden: int = 3
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: Array) -> Array:
        n_el = x.shape[-2]
        n_dim = x.shape[-1]
        n_ion = self.ions.shape[-2]
        n_feature = n_ion * (n_dim + 1)
        resnet = build_mlp(
            [n_feature]*self.n_hidden + [self.n_orb], 
            residual=True, activation=self.activation, param_dtype=_t_real)
        # build input features
        diff_ei = diffmat(x, self.ions) # [..., n_el, n_ion, 3]
        r_ei = jnp.linalg.norm(diff_ei, axis=-1, keepdims=True) # [..., n_el, n_ion, 1]
        feature = jnp.concatenate([diff_ei, r_ei], axis=-1) # [..., n_el, n_ion, 4]
        feature = feature.reshape(*feature.shape[:-2], n_feature) # [..., n_el, (n_ion * 4)]
        # return the result from MLP
        return resnet(feature) # [..., n_el, n_orb]


class Slater(nn.Module):
    r"""Slater determinant from single particle orbitals
    
    Separate the electrons into different spins and calculate orbitals for both.
    if full_det is True, use one large determinant. Otherwise use two small ones.
    Return sign and log of determinant when called.
    """

    ions: Array
    elems: Array
    spin: Optional[int] = None # difference between alpha and beta spin
    full_det: bool = True
    orbital_type: str = "simple"
    orbital_args: dict = _field(default_factory=dict)

    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        n_el = x.shape[-2]
        if self.spin is None:
            n_up, n_dn = n_el//2, n_el - n_el//2
        else:
            n_up = (n_el + self.spin) // 2
            n_dn = n_up - self.spin
        assert n_up + n_dn == n_el
        
        if self.orbital_type == "simple":
            OrbCls = SimpleOrbital
        else:
            raise ValueError("unsupported orbital type")
        orb_up_fn = OrbCls(self.ions, (n_el if self.full_det else n_up), **self.orbital_args)
        orb_dn_fn = OrbCls(self.ions, (n_el if self.full_det else n_dn), **self.orbital_args)

        orb_up = orb_up_fn(x[..., :n_up, :]) # [..., n_up, n_el|n_up]
        orb_dn = orb_dn_fn(x[..., n_up:, :]) # [..., n_up, n_el|n_dn]

        if self.full_det:
            orb_full = jnp.concatenate([orb_up, orb_dn], axis=-2)
            return jnp.linalg.slogdet(orb_full)
        else:
            sign_up, ldet_up = jnp.linalg.slogdet(orb_up)
            sign_dn, ldet_dn = jnp.linalg.slogdet(orb_dn)
            return sign_up * sign_dn, ldet_up + ldet_dn


class ProductModel(nn.Module):
    r"""Pruduct of multiple model results.
    
    Assuming the models returns in log scale. 
    The signature of each submodel can either be pure: x -> log(f(x)) 
    or with sign: x -> sign(f(x)), log(|f(x)|).
    The model will return sign if any of its submodels returns sign.
    """

    submodels: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x: Array) -> Union[Array, Tuple[Array, Array]]:
        sign = 1.
        logf = 0.
        with_sign = True # False will make the sign optional

        for model in self.submodels:
            result = model(x)
            if isinstance(result, tuple):
                sign *= result[0]
                logf += result[1]
                with_sign = True
            else:
                logf += result
        
        if with_sign:
            return sign, logf
        else:
            return logf


def build_jastrow_slater(ions, elems, spin=None, 
        full_det=True, orbital_type="simple", orbital_args=None):
    orbital_args = orbital_args or {}
    jastrow = Jastrow(ions, elems)
    slater = Slater(ions, elems, spin, full_det, orbital_type, orbital_args)
    model = ProductModel([jastrow, slater])
    return model
    