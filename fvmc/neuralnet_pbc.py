import dataclasses
from typing import Optional, Tuple, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp

from .utils import (Array, _t_real, adaptive_residual, build_mlp,
                    collect_elems, displace_matrix, fix_init, log_linear_exp,
                    parse_activation, wrap_complex_linear)
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
    dist = jnp.linalg.norm(
        d_hsin + jnp.eye(n_p)[..., None], 
        keepdims=True, axis=-1) * (1.0 - jnp.eye(n_p)[..., None])
    freqs = jnp.arange(1, n_freq+1).reshape(-1, 1)
    radfreqs = 2 * jnp.pi * freqs
    d_asin = jnp.sin(radfreqs * d_frac[:,:,None,:])
    d_acos = jnp.cos(radfreqs * d_frac[:,:,None,:])
    d_freq = jnp.concatenate([
        d_asin, d_acos
    ], axis=-2).reshape(n_p, n_p, -1)
    h2 = jnp.concatenate([
        d_freq, dist
    ], axis=-1)
    return h1, h2, dist


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
    spins: tuple[int, int]
    cell: Array
    n_out: int
    n_k: Optional[int] = None
    close_shell: bool = True
    use_complex: bool = False
    pair_type: str = 'product'

    @nn.compact
    def __call__(self, h1, x):
        # check shapes
        n_up, n_dn = self.spins
        n_max = max(n_up, n_dn)
        n_el, n_d = x.shape
        assert n_up + n_dn == n_el
        h1 = h1[-n_el:]
        # prepare k grid
        n_k = self.n_k or n_el // 2
        recvec = jnp.linalg.inv(self.cell).T
        kpts = jnp.asarray(gen_kidx(n_d, n_k, self.close_shell))
        kvecs = 2 * jnp.pi * kpts @ recvec # [n_k, 3]
        # backflow x and inner product
        x_bf = x + nn.Dense(n_d, param_dtype=_t_real)(h1) # [n_el, 3]
        evlp = self.pair_map(kvecs, x_bf)
        # padding for unpaired spin (this shall be changed)
        base = jnp.ones((self.n_out, n_max, n_max), dtype=evlp.dtype)
        return base.at[:, :n_up, :n_dn].set(evlp)

    def pair_map(self, k, x):
        if self.pair_type.lower().startswith('prod'):
            return self._prod_pair(k, x)
        if self.pair_type.lower().startswith('diag'):
            return self._diag_pair(k, x)
        if self.pair_type.lower().startswith('gen'):
            return self._general_pair(k, x)
        raise ValueError(f'unknown pair_type: {self.pair_type}')

    def _prod_pair(self, k, x):
        n_up, n_dn = self.spins
        k_dot_x = x @ k.T # [n_el, n_k]
        linear_map = nn.Dense(self.n_out, False, param_dtype=_t_real)
        # pw_kx: [n_elec, (2*)n_k]
        if self.use_complex:
            pw_kx = jnp.exp(1j * k_dot_x)
            linear_map = wrap_complex_linear(linear_map)
        else:
            pw_kx = jnp.concatenate([jnp.sin(k_dot_x), jnp.cos(k_dot_x)], -1)
        # ev_1b: [n_elec, n_out]
        ev_1b = linear_map(pw_kx)
        ev_up, ev_dn = jnp.split(ev_1b, [n_up], axis=0)
        evlp = (ev_up[:, None, :] * ev_dn.conj())
        return evlp.transpose(2, 0, 1) # [n_out, n_up, n_dn]
    
    def _diag_pair(self, k, x):
        n_up, n_dn = self.spins
        x_up, x_dn = jnp.split(x, [n_up], axis=0)
        x_delta = x_up[:, None, :] - x_dn # [n_up, n_dn, 3]
        k_dot_dx = x_delta @ k.T # [n_up, n_dn, n_k]
        linear_map = nn.Dense(self.n_out, False, param_dtype=_t_real)
        # pw_kx: [n_up, n_dn, n_k]
        if self.use_complex:
            pw_kx = jnp.exp(1j * k_dot_dx)
            linear_map = wrap_complex_linear(linear_map)
        else:
            pw_kx = jnp.concatenate([jnp.sin(k_dot_dx), jnp.cos(k_dot_dx)], -1)
        # evlp: [n_up, n_dn, n_out]
        evlp = linear_map(pw_kx)
        return evlp.transpose(2, 0, 1) # [n_out, n_up, n_dn]
    
    def _general_pair(self, k, x):
        n_up, n_dn = self.spins
        n_f = k.shape[0] if self.use_complex else k.shape[0] * 2
        k_dot_x = x @ k.T # [n_el, n_k]
        linear_map = nn.Dense(self.n_out * n_f, False, param_dtype=_t_real,
                              kernel_init=nn.initializers.zeros)
        # pw_kx: [n_elec, (2*)n_k]
        if self.use_complex:
            pw_kx = jnp.exp(1j * k_dot_x)
            linear_map = wrap_complex_linear(linear_map)
        else:
            pw_kx = jnp.concatenate([jnp.sin(k_dot_x), jnp.cos(k_dot_x)], -1)
        ev_1b = (linear_map(pw_kx).reshape(n_up+n_dn, self.n_out, n_f)
                 + pw_kx[:, None, :])
        ev_up, ev_dn = jnp.split(ev_1b, [n_up], axis=0)
        # ev_pair: [n_out, n_up, n_dn, (2*)n_k]
        ev_pair = jnp.einsum('iak,jak->aijk', ev_up, ev_dn.conj())
        diag_v = self.param("v", nn.initializers.normal(0.01), (n_f, 1))
        evlp = (ev_pair @ diag_v if not self.use_complex else
                wrap_complex_linear(lambda x: x @ diag_v)(ev_pair))
        return evlp.squeeze(-1) # [n_out, n_up, n_dn]


class GeminalMap(nn.Module):
    spins: tuple[int, int]
    cell: Array
    determinants: int

    @nn.compact
    def __call__(self, h1, h2=None):
        n_det = self.determinants
        n_up, n_dn = self.spins
        n_sdiff = abs(n_up - n_dn)
        n_el, h1_size = h1.shape
        assert n_up + n_dn == n_el
        # single orbital
        sorbitals = nn.Dense(h1_size, param_dtype=_t_real)(h1) # [n_elec, h1_size]
        sorb_up, sorb_dn = jnp.split(sorbitals, [n_up], axis=0)
        # pad ones for smaller spin
        if n_up < n_dn:
            sorb_up = jnp.concatenate(
                [sorb_up, jnp.ones((n_sdiff, h1_size))], axis=0)
        elif n_up > n_dn:
            sorb_dn = jnp.concatenate(
                [sorb_dn, jnp.ones((n_sdiff, h1_size))], axis=0)
        # make geminals
        w = self.param("w", nn.initializers.normal(0.01), (h1_size, n_det))
        # make it kfac recognizable
        # pair_orbs = jnp.einsum('lk,il,jl->kij', w, sorb_up, sorb_dn.conj())
        pair_orbs = ((sorb_up[:,None,:] * sorb_dn.conj()) @ w).transpose(2,0,1)
        return pair_orbs + 1


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
    identical_h1_update: bool = False
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
                identical_h1_update=self.identical_h1_update,
                identical_h2_update=self.identical_h2_update
            )
            h1, h2 = flayer(h1, h2)

        h1n, h1e = jnp.split(h1, [n_nucl], axis=0)
        h2e = h2[-n_elec:, -n_elec:]

        geminal_map = GeminalMap(self.spins, self.cell, 
                                 self.determinants)
        geminals = geminal_map(h1e, h2e)

        envelope_map = PbcEnvelope(self.spins, self.cell,
                                   self.determinants, **self.envelope)
        envelopes = envelope_map(h1e, x)

        dweight_map = nn.Dense(self.determinants, param_dtype=_t_real)
        det_weights = dweight_map(h1n.mean(0, keepdims=True)).T # [n_det, 1]
        dets = geminals * envelopes
        signs, logdets = jnp.linalg.slogdet(dets)
        sign, logpsi = log_linear_exp(signs, logdets, det_weights, axis=0)
        sign, logpsi = sign[0], logpsi[0]

        jastrow = build_mlp([h1.shape[-1]] * self.jastrow_layers + [1],
            residual=True, activation=self.activation, 
            rescale=self.rescale_residual, param_dtype=_t_real)
        jastrow_weight = self.param(
            "jastrow_weights", nn.initializers.zeros, ())
        logpsi += jastrow_weight * jastrow(h1e).mean()

        # electron-electron cusp condition (not in use)
        # cusp = ElectronCusp((n_up, n_dn))
        # logpsi += cusp(d_ee=dmat[n_nucl:, n_nucl:, -1])
        
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