import dataclasses
from typing import Optional, Tuple, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp

from .utils import (Array, _t_real, adaptive_residual, build_mlp,
                    collect_elems, displace_matrix, fix_init, log_linear_exp,
                    parse_activation)
from .wavefunction import FullWfn
from .neuralnet import FermiLayer, ElectronCusp


def raw_features_pbc(r, x, latvec, n_freq):
    n_elec = x.shape[0]
    n_nucl = r.shape[0]
    n_p = n_nucl + n_elec
    # use both nuclei and electron positions
    pos = jnp.concatenate([r, x], axis=0)
    # initial h1 is empty, trick to avoid error in kfac
    h1 = pos[:, :1] * 0
    # h2 for pbc handling
    invvec = jnp.linalg.inv(latvec)
    disp = displace_matrix(pos, pos)
    d_frac = disp @ invvec
    d_hsin = jnp.sin(jnp.pi * d_frac) @ latvec/jnp.pi
    freqs = jnp.arange(n_freq).reshape(-1, 1)
    d_asin = jnp.sin(2*jnp.pi * freqs * d_frac[:,:,None,:]) @ latvec/jnp.pi
    d_acos = jnp.cos(2*jnp.pi * freqs * d_frac[:,:,None,:]) @ latvec/jnp.pi
    d_freq = jnp.concatenate([
        d_asin, d_acos
    ], axis=-2).reshape(n_p, n_p, -1)
    # calc dist and rescaling
    dist = jnp.linalg.norm(
        d_hsin + jnp.eye(n_p)[..., None],
        keepdims=True,
        axis=-1
    )
    h2_scaling = jnp.log(1 + dist) / dist
    dmat = jnp.concatenate([
        d_freq, 
        dist * (1.0 - jnp.eye(n_p)[..., None])
    ], axis=-1)
    h2 = dmat * h2_scaling
    return h1, h2, dmat


def gen_kidx(n_d, n_k, close_shell=True):
    # n_d is spacial dimension
    # n_k is number of k points
    n_max = int(onp.ceil((n_k/2) ** (1/n_d)))
    grid = onp.arange(-n_max, n_max+1, dtype=int)
    mesh = onp.stack(onp.meshgrid(*([grid] * n_d), indexing='ij'), axis=-1)
    kall = mesh.reshape(-1, n_d)
    k2 = (kall ** 2).sum(-1)
    sidx = onp.argsort(k2)
    if not close_shell:
        return kall[sidx[:n_k]]
    else:
        shell_idx = onp.nonzero(k2 <= k2[sidx[n_k-1]])
        return kall[shell_idx]
    

class PbcEnvelope(nn.Module):
    cell: Array
    n_out: int
    n_k: Optional[int] = None
    close_shell: bool = True
    use_complex: bool = False

    @nn.compact
    def __call__(self, h1, x):
        # check shapes
        n_el, n_d = x.shape
        h1 = h1[-n_el:]
        # prepare k grid
        n_k = self.n_k or n_el // 2
        recvec = jnp.linalg.inv(self.cell).T
        kpts = jnp.asarray(gen_kidx(n_d, n_k, self.close_shell))
        kvecs = 2 * jnp.pi * kpts @ recvec # [n_k, 3]
        # backflow x and inner product
        x_bf = nn.Dense(n_d)(h1) + x # [n_el, 3]
        k_dot_x = x_bf @ kvecs.T # [n_el, n_k]
        # potential flip k for spin down
        # k_dot_x = k_dot_x.at[-n_dn:].multiply(-1)
        if self.use_complex:
            eikx = jnp.exp(1j * k_dot_x)
            return nn.Dense(self.n_out, use_bias=False)(eikx)
        else:
            sinkx = jnp.sin(k_dot_x)
            coskx = jnp.cos(k_dot_x)
            return (nn.Dense(self.n_out, use_bias=False)(sinkx) 
                  + nn.Dense(self.n_out, use_bias=False)(coskx))
        

class GeminalMap(nn.Module):
    spins: tuple[int, int]
    cell: Array
    determinants: int
    envelope: dict = dataclasses.field(default_factory=dict)

    @nn.compact
    def __call__(self, h1, x):
        n_det = self.determinants
        n_up, n_dn = self.spins
        n_elec = n_up + n_dn
        n_sdiff = abs(n_up - n_dn)
        h1_size = h1.shape[-1]
        h1_el = h1[-n_elec:]
        # single orbital
        sorbitals = nn.Dense(h1_size)(h1_el) # [n_elec, h1_size]
        sorb_up, sorb_dn = jnp.split(sorbitals, [n_up], axis=0)
        # single envelope
        envelopes = PbcEnvelope(self.cell, n_det, **self.envelope)(h1_el, x)
        evlp_up, evlp_dn = jnp.split(envelopes, [n_up], axis=0)
        # pad ones for smaller spin
        if n_up < n_dn:
            sorb_up, evlp_up = [jnp.concatenate([
                arr, jnp.ones((n_sdiff, arr.shape[-1]))
            ], axis=0) for arr in (sorb_up, evlp_up)]
        elif n_up > n_dn:
            sorb_dn, evlp_dn = [jnp.concatenate([
                arr, jnp.ones((n_sdiff, arr.shape[-1]))
            ], axis=0) for arr in (sorb_dn, evlp_dn)]
        # make geminals
        w = self.param("w", nn.initializers.normal(0.01), (n_det, h1_size))
        pair_orbs = jnp.einsum('kl,il,jl->kij', w, sorb_up, sorb_dn.conj())
        pair_envs = jnp.einsum('ik,jk->kij', evlp_up, evlp_dn.conj())
        return pair_orbs * pair_envs


class FermiNetPbc(FullWfn):
    elems: Sequence[int]
    spins: tuple[int, int]
    cell: Array
    raw_freq: int = 5
    hidden_dims: Sequence[Tuple[int, int]] = ((64, 16),)*4
    determinants: int = 4
    envelope: dict = dataclasses.field(default_factory=dict)
    activation: str = "gelu"
    rescale_residual: bool = True
    type_embedding: int = 5
    jastrow_layers: int = 3
    spin_symmetry: bool = True
    identical_h2_update: bool = False

    @nn.compact
    def __call__(self, r: Array, x: Array) -> Array:
        n_elec = x.shape[-2]
        n_nucl = r.shape[-2]
        _, elem_sec = collect_elems(self.elems)
        n_up, n_dn = self.spins
        assert sum(elem_sec) == n_nucl
        assert n_up + n_dn == n_elec
        split_sec = onp.asarray([*elem_sec, n_up, n_dn])

        h1, h2, dmat = raw_features_pbc(r, x, self.cell, self.raw_freq)
        if self.type_embedding > 0:
            type_embd = self.param("type_embedding", 
                nn.initializers.normal(1.0), 
                (len(split_sec), self.type_embedding), _t_real)
            h1 = jnp.concatenate([
                h1, jnp.repeat(type_embd, split_sec, axis=0)
            ], axis=1)

        for ii, (sdim, pdim) in enumerate(self.hidden_dims):
            flayer = FermiLayer(
                single_size=sdim,
                pair_size=pdim,
                split_sec=split_sec,
                activation='tanh' if ii==0 else self.activation,
                rescale_residual=self.rescale_residual,
                spin_symmetry=self.spin_symmetry,
                identical_h2_update=self.identical_h2_update
            )
            h1, h2 = flayer(h1, h2)

        geminal_map = GeminalMap(self.spins, self.cell, 
                                 self.determinants, self.envelope)
        geminals = geminal_map(h1[-n_elec:], x)

        signs, logdets = jnp.linalg.slogdet(geminals)
        det_weights = self.param(
            "det_weights", nn.initializers.ones, (self.determinants, 1))
        sign, logpsi = log_linear_exp(signs, logdets, det_weights, axis=0)
        sign, logpsi = sign[0], logpsi[0]

        jastrow = build_mlp([h1.shape[-1]] * self.jastrow_layers + [1],
            residual=True, activation=self.activation, 
            rescale=self.rescale_residual, param_dtype=_t_real)
        jastrow_weight = self.param(
            "jastrow_weights", nn.initializers.zeros, ())
        logpsi += jastrow_weight * jastrow(h1).mean()
        # electron-electron cusp condition
        cusp = ElectronCusp((n_up, n_dn))
        logpsi += cusp(d_ee=dmat[n_nucl:, n_nucl:, -1])
        
        return sign, logpsi
    

class NucleiGaussianSlaterPbc(FullWfn):
    r"""Gaussian for nuclei wavefunctions with Slater determinant exchange
    
    The wavefunction is given by Det_ij{ exp[-(r_i - r0_j)^2 / (2 * sigma_j^2)] }
    """

    cell: Array
    init_r0: Array
    init_sigma: Array

    @nn.compact
    def __call__(self, r: Array, x: Array) -> Tuple[Array, Array]:
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