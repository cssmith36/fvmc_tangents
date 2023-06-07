import abc
from dataclasses import field as _field
from typing import Optional, Sequence, Tuple, Union

import jax
from flax import linen as nn
from jax import numpy as jnp

from fvmc.utils import Array

from .utils import (Array, _t_real, build_mlp, cdist, displace_matrix, fix_init,
                    parse_spin, pdist)


def log_prob_from_model(model: nn.Module):
    if isinstance(model, FullWfn): # combine r, x into a tuple conf = (r, x)
        return lambda p, conf: 2 * model.apply(p, *conf)[1]
    elif isinstance(model, ElecWfn): # electron only case
        return lambda p, x: 2 * model.apply(p, x)[1]
    else: # fall back for any model
        # raise TypeError(f"unsupoorted model type {type(model)}")
        return lambda p, *args, **kwargs: 2 * model.apply(p, *args, **kwargs)[1]


class FullWfn(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, r: Array, x: Array) -> Tuple[Array, Array]:
        """Take nuclei position r and electron position x, return sign and log|psi|"""
        raise NotImplementedError
    

class ElecWfn(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Take only the electron position x, return sign and log|psi|"""
        raise NotImplementedError


class FixNuclei(ElecWfn):
    r"""Module warpper that fix the nuclei positions for a full model
    
    This class takes a full wavefunction model f(r,x) of r (nuclei) and x (electrons)
    and the fixed nuclei positions r_0, and return a new model which only depends on x.
    Think it as a partial warpper that works on nn.Module
    """
    model: FullWfn
    nuclei: Array

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        return self.model(r=self.nuclei, x=x)


class ProductModel(FullWfn):
    r"""Pruduct of multiple model results.
    
    Assuming the models returns in log scale. 
    The signature of each submodel can either be pure: x -> log(f(x)) 
    or with sign: x -> sign(f(x)), log(|f(x)|).
    The model will return sign if any of its submodels returns sign.
    """

    submodels: Sequence[nn.Module]

    @nn.compact
    def __call__(self, r:Array, x: Array) -> Tuple[Array, Array]:
        sign = 1.
        logf = 0.
        with_sign = True # False will make the sign optional

        for model in self.submodels:
            result = model(r, x)
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
    def __call__(self, r: Array, x: Array) -> Array:
        # calculate initial scale, so that it returns 0 if all electrons are on nuclei
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
    def __call__(self, r: Array, x: Array) -> Array:
        n_el = x.shape[-2]
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

    spin: Optional[int] = None # difference between alpha and beta spin
    full_det: bool = True
    orbital_type: str = "simple"
    orbital_args: dict = _field(default_factory=dict)

    @nn.compact
    def __call__(self, r: Array, x: Array) -> Tuple[Array, Array]:
        n_el = x.shape[-2]
        n_up, n_dn = parse_spin(n_el, self.spin)
        
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
        

class NucleiGaussian(FullWfn):
    r"""Gaussian for nuclei wavefunctions, centered on trainable sites
    
    The log wavefunction is given by - \sum_i (r_i - r0_i)^2 / (2 * sigma_i^2)
    """

    init_r0: Array
    init_sigma: Array

    @nn.compact
    def __call__(self, r: Array, x: Array) -> Tuple[Array, Array]:
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
    def __call__(self, r: Array, x: Array) -> Tuple[Array, Array]:
        del x
        # r0: [n_nucl, 3], sigma: [n_nucl, 1]
        r0 = self.param("r0", fix_init, self.init_r0, _t_real)
        sigma = self.param("sigma", fix_init, 
                           jnp.reshape(self.init_sigma, (-1, 1)), _t_real)
        # exps: [n_nucl, n_nucl]
        exps = jnp.exp(-0.5 * jnp.sum(((r[:, None] - r0) / sigma)**2, -1))
        return jnp.linalg.slogdet(exps)


def build_jastrow_slater(elems, nuclei, spin=None, dynamic_nuclei=False,
        full_det=True, orbital_type="simple", orbital_args=None):
    orbital_args = orbital_args or {}
    jastrow = SimpleJastrow(elems)
    slater = SimpleSlater(spin, full_det, orbital_type, orbital_args)
    mlist = [jastrow, slater]
    if dynamic_nuclei:
        mlist.append(NucleiGaussianSlater(nuclei, 0.1))
    model = ProductModel(mlist)
    # elec_model = FixNuclei(model, nuclei)
    return model
