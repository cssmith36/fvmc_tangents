import dataclasses
from typing import Optional, Sequence, Tuple

import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp

from ..utils import (Array, ElecConf, NuclConf, _t_real, build_mlp,
                     collect_elems, displace_matrix, ensure_no_spin, gen_kidx,
                     log_linear_exp, wrap_complex_linear)
from .base import FullWfn
from .neuralnet import ElectronCusp, FermiLayer


def dist_features_pbc(pos, latvec, frac_dist=False, keepdims=True):
    n = len(pos)
    # orthorhombic minimum-image displacement
    invvec = jnp.linalg.inv(latvec)
    pos_frac = pos @ invvec
    d_frac = displace_matrix(pos_frac, pos_frac)
    d_frac = d_frac - jnp.rint(d_frac)
    # sin wrap displacement
    d_hsin = jnp.sin(jnp.pi * d_frac)  # output \in [0, 1)
    if not frac_dist:
        d_hsin = d_hsin @ (latvec / jnp.pi)
    # distance (pad diagonal)
    dist = jnp.linalg.norm(
        d_hsin + jnp.eye(n)[..., None],
        keepdims=keepdims, axis=-1)
    # zero diagonal
    if keepdims:
        dist = dist * (1.0 - jnp.eye(n)[..., None])
    else:
        dist = dist * (1.0 - jnp.eye(n))
    return d_frac, d_hsin, dist


def raw_features_pbc(r, x, latvec, n_freq, frac_dist=False):
    n_elec = x.shape[0]
    n_nucl = r.shape[0]
    n_p = n_nucl + n_elec
    # use both nuclei and electron positions
    pos = jnp.concatenate([r, x], axis=0)
    # initial h1 is empty, trick to avoid error in kfac
    h1 = pos[:, :1] * 0
    # h2 for pbc handling
    d_frac, _, dist = dist_features_pbc(pos, latvec,
        frac_dist=frac_dist, keepdims=True)
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


class PbcEnvelope(nn.Module):
    spins: tuple[int, int]
    cell: Array
    n_out: int
    n_k: Optional[int] = None
    close_shell: bool = True
    use_complex: bool = True
    pair_type: str = 'general'

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
        w = self.param("w", nn.initializers.normal(1), (h1_size, n_det)) + 1
        # make it kfac recognizable
        # pair_orbs = jnp.einsum('lk,il,jl->kij', w, sorb_up, sorb_dn.conj())
        pair_orbs = ((sorb_up[:,None,:] * sorb_dn.conj()) @ w).transpose(2,0,1)
        return pair_orbs + 1


class FermiNetPbc(FullWfn):
    elems: Sequence[int]
    spins: tuple[int, int]
    cell: Array
    raw_freq: int = 5
    determinants: int = 4
    hidden_dims: Sequence[Tuple[int, int]] = ((64, 16),)*4
    activation: str = "gelu"
    rescale_residual: bool = True
    envelope: dict = dataclasses.field(default_factory=dict)
    fermilayer: dict = dataclasses.field(default_factory=dict)
    type_embedding: int = 5
    jastrow_layers: int = 3
    nuclei_module: Optional[FullWfn] = None

    @nn.compact
    def __call__(self, r: NuclConf, x: ElecConf) -> Array:
        x = ensure_no_spin(x)
        n_elec = x.shape[-2]
        n_nucl = r.shape[-2]
        _, elem_sec = collect_elems(self.elems)
        n_up, n_dn = self.spins
        assert sum(elem_sec) == n_nucl
        assert n_up + n_dn == n_elec
        split_sec = np.asarray([*elem_sec, n_up, n_dn])

        h1, h2, dmat = raw_features_pbc(r, x, self.cell, self.raw_freq)
        if self.type_embedding > 0:
            type_embd = self.param("type_embedding",
                nn.initializers.normal(1.0),
                (len(split_sec), self.type_embedding), _t_real)
            h1 = jnp.concatenate([
                h1, jnp.repeat(type_embd, split_sec, axis=0)
            ], axis=1)

        for ii, (sdim, pdim) in enumerate(self.hidden_dims):
            flargs = ({**self.fermilayer, "h2_convolution": False}
                      if ii == 0 else self.fermilayer)
            flayer = FermiLayer(
                single_size=sdim,
                pair_size=pdim,
                split_sec=split_sec,
                activation='tanh' if ii==0 else self.activation,
                rescale_residual=self.rescale_residual,
                **flargs
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
        dw_in = (h1n if n_nucl > 0 else h1e).mean(0, keepdims=True)
        det_weights = dweight_map(dw_in).T # [n_det, 1]
        dets = geminals * envelopes
        signs, logdets = jnp.linalg.slogdet(dets)
        sign, logpsi = log_linear_exp(signs, logdets, det_weights, axis=0)
        sign, logpsi = sign[0], logpsi[0]

        jastrow = build_mlp([h1.shape[-1]] * self.jastrow_layers + [1],
            residual=True, activation=self.activation, last_bias=False,
            rescale=self.rescale_residual, param_dtype=_t_real)
        jastrow_weight = self.param(
            "jastrow_weights", nn.initializers.zeros, ())
        logpsi += jastrow_weight * jastrow(h1e).mean()

        # nuclei module with backflowed r
        if self.nuclei_module is not None:
            r_bf = r + nn.Dense(r.shape[-1], param_dtype=_t_real,
                                kernel_init=nn.initializers.zeros)(h1n)
            nuc_sign, nuc_logpsi = self.nuclei_module(r_bf, x)
            sign *= nuc_sign
            logpsi += nuc_logpsi

        # electron-electron cusp condition (not in use)
        # cusp = ElectronCusp((n_up, n_dn))
        # logpsi += cusp(d_ee=dmat[n_nucl:, n_nucl:, -1])

        return sign, logpsi
