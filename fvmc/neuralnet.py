import functools
from dataclasses import field
from typing import Any, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp

from .wavefunction import FullWfn
from .utils import (Array, adaptive_residual, build_mlp, cdist, diffmat,
                    fix_init, log_linear_exp, parse_activation, parse_spin,
                    pdist)

# for all functions, we use the following convention:
# r for atom positions
# x for electron positions
# elems for the nuclear charges (elements)


def raw_features(r, x):
    n_elec = x.shape[0]
    n_atom = r.shape[0]
    # Electron-atom distances
    d_ei = diffmat(x, r)
    d_ei_norm = jnp.linalg.norm(d_ei, keepdims=True, axis=-1)
    d_ei = jnp.concatenate([d_ei, d_ei_norm], axis=-1)
    h1_scaling = jnp.log(1 + d_ei_norm) / d_ei_norm
    h1 = d_ei * h1_scaling # [n_elec, n_atom, 4]
    # Electron-electron distances
    d_ee = diffmat(x, x)
    d_ee_norm = jnp.linalg.norm(
        d_ee + jnp.eye(n_elec)[..., None],
        keepdims=True,
        axis=-1
    )
    h2_scaling = jnp.log(1 + d_ee_norm) / d_ee_norm
    d_ee = jnp.concatenate([d_ee, 
        d_ee_norm * (1.0 - jnp.eye(n_elec)[..., None])], axis=-1)
    h2 = d_ee * h2_scaling # [n_elec, n_elec, 4]
    return h1, h2, d_ei, d_ee


class Atom_Embedding(nn.Module):
    embedding_size: int
    activation: str = 'tanh'
    weight: Optional[Array] = None
    bias: Optional[Array] = None

    @nn.compact
    def __call__(self, h1: Array) -> Array:
        n_elec, n_atom, _ = h1.shape
        n_embd = self.embedding_size
        actv_fn = parse_activation(self.activation, rescale=True)
        # initialize weight and bias if not given as input
        weight = (self.param(
                'weight',
                nn.initializers.normal(1/onp.sqrt(4)),
                (n_atom, 4, n_embd))
            if self.weight is None else self.weight)
        bias = (self.param(
                'bias',
                nn.initializers.normal(1.),
                (n_atom, n_embd))
            if self.bias is None else self.bias)
        # do the calculation
        assert weight.shape == (n_atom, 4, n_embd)
        assert bias.shape == (n_atom, n_embd)
        h1 = jnp.einsum('nmi,mio->nmo', h1, weight) + bias
        h1 = actv_fn(h1) # TODO: check if this is better to use layer norm
        h1 = h1.mean(axis=1)
        return h1


def aggregate_features(h1, h2, n_elec, absolute_spin):
    n_elec = onp.array(n_elec)
    assert n_elec.sum() == h1.shape[0]
    n_up, n_dn = n_elec
    # Single input
    h2_mean = jnp.stack(
        [
            h2_spin.mean(axis=1)
            for h2_spin in jnp.split(h2, n_elec[:1], axis=1) 
            if h2_spin.size > 0
        ], axis=-2)
    if not absolute_spin:
        h2_mean = h2_mean.at[n_up:].set(h2_mean[n_up:, (1, 0)])
    one_in = jnp.concatenate([h1, h2_mean.reshape(h1.shape[0], -1)], axis=-1)
    # Global input
    h1_up, h1_dn = jnp.split(h1, n_elec[:1], axis=0)
    all_up, all_dn = h1_up.mean(0), h1_dn.mean(0)
    if absolute_spin:
        all_in = jnp.array([[all_up, all_dn], [all_up, all_dn]])
    else:
        all_in = jnp.array([[all_up, all_dn], [all_dn, all_up]])
    all_in = all_in.reshape(2, -1)
    return one_in, all_in


class FermiLayer(nn.Module):
    single_size: int
    pair_size: int
    n_elec: Tuple[int, int]
    activation: str = "gelu"
    rescale_residual: bool = True
    absolute_spin: bool = False
    update_pair_independent: bool = False

    @nn.compact
    def __call__(self, h1, h2):
        actv_fn = parse_activation(self.activation, rescale=self.rescale_residual)

        # Single update
        one_in, all_in = aggregate_features(h1, h2, self.n_elec, self.absolute_spin)
        # per electron contribution
        one_new = nn.Dense(self.single_size)(one_in)
        # global contribution
        all_new = nn.Dense(self.single_size, use_bias=False)(all_in) # [2, n_single]
        all_new = all_new.repeat(onp.array(self.n_elec), axis=0) # broadcast to both spins
        # combine both of them to get new h1
        h1_new = (one_new + all_new) #/ jnp.sqrt(2.0)
        h1 = adaptive_residual(h1, actv_fn(h1_new), rescale=self.rescale_residual)
        
        # Pairwise update
        if self.update_pair_independent:
            h2_new = nn.Dense(self.pair_size)(h2)
        else:
            u, d = jnp.split(h2, self.n_elec[:1], axis=0)
            uu, ud = jnp.split(u, self.n_elec[:1], axis=1)
            du, dd = jnp.split(d, self.n_elec[:1], axis=1)
            same = nn.Dense(self.pair_size)
            diff = nn.Dense(self.pair_size)
            h2_new = jnp.concatenate([
                jnp.concatenate([same(uu), diff(ud)], axis=1),
                jnp.concatenate([diff(du), same(dd)], axis=1),
            ], axis=0)
        if h2.shape != h2_new.shape: # fitst layer
            h2 = jnp.tanh(h2_new)
        else:
            h2 = adaptive_residual(h2, actv_fn(h2_new), rescale=self.rescale_residual)
        return h1, h2


class IsotropicEnvelope(nn.Module):
    out_size: int
    determinants: int
    softplus: bool = True
    sigma: Optional[Array] = None
    pi: Optional[Array] = None

    @nn.compact
    def __call__(self, d_ei): 
        assert d_ei.ndim <= 3
        d_ei = jnp.atleast_3d(d_ei)[:, :, -1:, None] # [n_elec, n_atom, 1, 1]
        n_atom = d_ei.shape[1]
        param_shape = (n_atom, self.out_size, self.determinants)
        sigma = (self.param('sigma', nn.initializers.ones, param_shape)
            if self.sigma is None else self.sigma)
        pi = (self.param('pi', nn.initializers.ones, param_shape)
            if self.pi is None else self.pi)
        assert sigma.shape == pi.shape == param_shape
        if self.softplus:
            sigma = nn.softplus(sigma)
            pi = nn.softplus(pi)
        return jnp.sum(pi * jnp.exp(-sigma * d_ei), axis=1)
    

class OrbitalMap(nn.Module):
    n_elec: Tuple[int, int]
    determinants: int
    full_det: bool = True
    share_weights: bool = False
    envelope: Optional[Tuple[nn.Module, nn.Module]] = None

    @nn.compact
    def __call__(self, h1, d_ei):
        # h_one is [n_elec, n_desc]
        # d_ei is [n_elec, n_atom, 1]
        n_el = h1.shape[0]
        n_det = self.determinants
        assert sum(self.n_elec) == n_el
        evlps = (None, None) if self.envelope is None else self.envelope

        # make orbital from h1 and envelope
        def orbital_fn(h, d, n_orb, envelope=None):
            n_param = n_orb * n_det
            dense = nn.Dense(n_param)
            envelope = (IsotropicEnvelope(n_orb, self.determinants)
                        if envelope is None else envelope)
            assert envelope.out_size == n_orb
            # Actual orbital function
            return dense(h).reshape(n_el, n_orb, n_det) * envelope(d) 

        # Case destinction for weight sharing 
        if self.share_weights:
            uu, dd = jnp.split(orbital_fn(h1, d_ei, max(self.n_elec), evlps[0]), 
                               self.n_elec[:1], axis=0)
            ud, du = jnp.split(orbital_fn(h1, d_ei, max(self.n_elec), evlps[1]), 
                               self.n_elec[:1], axis=0)
            if self.full_det:
                orbitals = (jnp.concatenate([
                    jnp.concatenate([uu[:, :self.n_elec[0]], ud[:, :self.n_elec[1]]], axis=1),
                    jnp.concatenate([du[:, :self.n_elec[0]], dd[:, :self.n_elec[1]]], axis=1),
                ], axis=0),)
            else:
                orbitals = (uu[:, :self.n_elec[0]], dd[:, :self.n_elec[1]])
        else:
            h_by_spin = jnp.split(h1, self.n_elec[:1], axis=0)
            d_by_spin = jnp.split(d_ei, self.n_elec[:1], axis=0)
            orbitals = tuple(
                orbital_fn(h, d, no, ev)
                for h, d, no, ev in zip(
                    h_by_spin,
                    d_by_spin,
                    (sum(self.n_elec),)*2 if self.full_det else self.n_elec,
                    evlps)
                )
            if self.full_det:
                orbitals = (jnp.concatenate(orbitals, axis=0),)
        return tuple(o.transpose(2, 0, 1) for o in orbitals)


class ElectronCusp(nn.Module):
    n_elec: tuple[int, int]
    
    @nn.compact
    def __call__(self, d_ee: Array) -> Array:
        w_para, w_anti = self.param('weight', fix_init, [1e-2]*2)
        a_para, a_anti = self.param('alpha', fix_init, [1.]*2)
        d_ee = jnp.atleast_3d(d_ee)[..., -1]
        uu, ud, du, dd = [
            s
            for split in jnp.split(d_ee, self.n_elec[:1], axis=0)
            for s in jnp.split(split, self.n_elec[:1], axis=1)
        ]
        same = jnp.concatenate([uu.reshape(-1), dd.reshape(-1)])
        diff = jnp.concatenate([ud.reshape(-1), du.reshape(-1)])
        result = -(1/4) * w_para * (a_para**2 / (a_para + same)).sum()
        result += -(1/2) * w_anti * (a_anti**2 / (a_anti + diff)).sum()
        return result


class FermiNet(FullWfn):
    spin: int = None
    hidden_dims: Sequence[Tuple[int, int]] = ((64, 16),)*4
    determinants: int = 16
    full_det: bool = True
    activation: str = "gelu"
    rescale_residual: bool = True
    jastrow_layers: int = 3
    embedding_size: Optional[int] = 32
    absolute_spins: bool = False
    update_pair_independent: bool = False

    @nn.compact
    def __call__(self, r: Array, x: Array) -> Array:
        n_elec = x.shape[-2]
        n_atom = r.shape[-2]
        n_elec_spin = parse_spin(n_elec, self.spin)
        raw_h1, h2, d_ei, d_ee = raw_features(r, x)

        embd_size = self.embedding_size or self.hidden_dims[0][0]
        atom_embedding = Atom_Embedding(embd_size)
        h1 = atom_embedding(raw_h1)

        for sdim, pdim in self.hidden_dims:
            flayer = FermiLayer(sdim, pdim, n_elec_spin, 
                self.activation, self.rescale_residual, 
                self.absolute_spins, self.update_pair_independent)
            h1, h2 = flayer(h1, h2)
        
        orbital_map = OrbitalMap(n_elec_spin, 
            self.determinants, self.full_det, not self.absolute_spins)
        orbitals = orbital_map(h1, d_ei) # tuple of [n_det, n_orb, n_orb]

        signs, logdets = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0),
            *jax.tree_map(jnp.linalg.slogdet, orbitals)) # [1 or 2, n_det]
        signs, logdets = signs.prod(0), logdets.sum(0)
        det_weights = self.param(
            "det_weights", nn.initializers.ones, (self.determinants, 1))
        sign, logpsi = log_linear_exp(signs, logdets, det_weights, axis=0)
        sign, logpsi = sign[0], logpsi[0]

        jastrow = build_mlp([h1.shape[-1]] * self.jastrow_layers + [1],
            residual=True, activation=self.activation, rescale=self.rescale_residual)
        jastrow_weight = self.param(
            "jastrow_weights", nn.initializers.zeros, ())
        cusp = ElectronCusp(n_elec_spin)
        logpsi += jastrow_weight * jastrow(h1).mean() + cusp(d_ee[...,-1])

        return sign, logpsi
