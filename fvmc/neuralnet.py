import functools
from dataclasses import field
from typing import Any, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp

from .utils import (Array, _t_real, adaptive_residual, build_mlp,
                    displace_matrix, fix_init, log_linear_exp, parse_activation,
                    parse_spin, collect_elems)
from .wavefunction import FullWfn

# for all functions, we use the following convention:
# r for atom positions
# x for electron positions
# elems for the nuclear charges (elements)


def raw_features(r, x):
    n_elec = x.shape[0]
    n_nucl = r.shape[0]
    n_particle = n_nucl + n_elec
    # use both nuclei and electron positions
    pos = jnp.concatenate([r, x], axis=0)
    # initial h1 is empty, trick to avoid error in kfac
    h1 = pos[:, :1] * 0
    # pair distances
    disp = displace_matrix(pos, pos)
    dist = jnp.linalg.norm(
        disp + jnp.eye(n_particle)[..., None],
        keepdims=True,
        axis=-1
    )
    h2_scaling = jnp.log(1 + dist) / dist
    dmat = jnp.concatenate([
        disp, 
        dist * (1.0 - jnp.eye(n_particle)[..., None])
    ], axis=-1)
    h2 = dmat * h2_scaling # [n_p, n_p, 4]
    return h1, h2, dmat


def aggregate_features(h1, h2, split_sec, spin_symmetry):
    split_sec = onp.asarray(split_sec)
    assert split_sec.ndim == 1 and len(split_sec) >= 2
    # last two sections are electrons with spin up and down
    *elem_sec, n_up, n_dn = split_sec
    n_nucl = sum(elem_sec)
    split_idx = onp.cumsum(split_sec)[:-1]
    # Single input
    h2_mean = jnp.stack(
        [
            h2_sec.mean(axis=1)
            for h2_sec in jnp.split(h2, split_idx, axis=1) 
            if h2_sec.size > 0
        ], axis=-2) # [n_particle, n_sec, n_desc]
    if spin_symmetry:
        h2_mean = h2_mean.at[-n_dn:, -2:].set(h2_mean[-n_dn:, (-1, -2)]) # switch spin
    nucl_one, elec_one = jnp.split(
        jnp.concatenate([h1, h2_mean.reshape(h1.shape[0], -1)], axis=-1),
        [n_nucl], axis=0)
    # Global input
    h1_mean = jnp.tile(jnp.stack(
        [
            h1_sec.mean(axis=0)
            for h1_sec in jnp.split(h1, split_idx, axis=0)
            if h1_sec.size > 0
        ], axis=-2), (3, 1, 1)) # [3, n_sec, n_desc], 3 for (nucl, e_up, e_dn)
    if spin_symmetry:
        h1_mean = h1_mean.at[-1, -2:].set(h1_mean[-1, (-1, -2)]) # switch spin
    nucl_all, elec_all = jnp.split(
        h1_mean.reshape(h1_mean.shape[0], -1), [1], axis=0)
    return nucl_one, nucl_all, elec_one, elec_all


class FermiLayer(nn.Module):
    single_size: int
    pair_size: int
    split_sec: Sequence[int]
    activation: str = "gelu"
    rescale_residual: bool = True
    spin_symmetry: bool = True
    identical_h2_update: bool = False

    @nn.compact
    def __call__(self, h1, h2):
        n_particle = sum(self.split_sec)
        assert n_particle == h1.shape[0] == h2.shape[0] == h2.shape[1]
        *elem_sec, n_up, n_dn = self.split_sec
        n_nucl = sum(elem_sec)
        actv_fn = parse_activation(self.activation, rescale=self.rescale_residual)

        # Single update
        nucl_one, nucl_all, elec_one, elec_all = aggregate_features(
            h1, h2, self.split_sec, self.spin_symmetry
        )

        # nuclei h1 part
        # per nucleus contribution
        n1_new = nn.Dense(self.single_size, param_dtype=_t_real)(nucl_one)
        # global contribution (all_new.shape == (2, n_single))
        na_new = nn.Dense(self.single_size, use_bias=False, param_dtype=_t_real)(nucl_all) 
        na_new = na_new.repeat(n_nucl, axis=0) # broadcast to all nuclei
        # combine both of them to get new h1
        nucl_new = n1_new + na_new #/ jnp.sqrt(2.0)

        # electron h1 part
        # per electron contribution
        e1_new = nn.Dense(self.single_size, param_dtype=_t_real)(elec_one)
        # global contribution (all_new.shape == (2, n_single))
        ea_new = nn.Dense(self.single_size, use_bias=False, param_dtype=_t_real)(elec_all) 
        ea_new = ea_new.repeat(onp.array([n_up, n_dn]), axis=0) # broadcast to both spins
        # combine both of them to get new h1
        elec_new = e1_new + ea_new #/ jnp.sqrt(2.0)

        h1_new = jnp.concatenate([nucl_new, elec_new], axis=0)
        assert h1_new.shape[:-1] == h1.shape[:-1]
        h1 = adaptive_residual(h1, actv_fn(h1_new), rescale=self.rescale_residual)
        
        # Pairwise update
        if self.identical_h2_update:
            h2_new = nn.Dense(self.pair_size, param_dtype=_t_real)(h2)
        else:
            pair_type = self._pair_type_idx()
            h2_new = jnp.zeros((n_particle, n_particle, self.pair_size), _t_real)
            for pt in range(pair_type.max()+1):
                ptidx = (pair_type == pt) 
                # doing different dense for different pair types
                h2_new = h2_new.at[ptidx, :].set(
                    nn.Dense(self.pair_size, param_dtype=_t_real)(h2[ptidx, :])
                )
        h2 = adaptive_residual(h2, actv_fn(h2_new), rescale=self.rescale_residual)
        return h1, h2
    
    @nn.nowrap
    def _pair_type_idx(self):
        # current pair type is 
        # n for nuclei, u for elec up, d for elec dn
        # (n u d)
        #  0 1 1 (n)
        #  1 2 3 (u)
        #  1 3 2 (d)
        n_particle = sum(self.split_sec)
        *elem_sec, n_up, n_dn = self.split_sec
        n_nucl = sum(elem_sec)
        slc_nucl, slc_elec = slice(0, n_nucl), slice(n_nucl, n_particle)
        slc_up, slc_dn = slice(n_nucl, n_nucl+n_up), slice(n_nucl+n_up, n_particle)
        pair_type = onp.zeros((n_particle, n_particle), int)
        pair_type[slc_nucl, slc_elec] = pair_type[slc_elec, slc_nucl] = 1
        pair_type[slc_up, slc_up] = pair_type[slc_dn, slc_dn] = 2
        pair_type[slc_up, slc_dn] = pair_type[slc_dn, slc_up] = 3
        return pair_type


class IsotropicEnvelope(nn.Module):
    out_size: int
    determinants: int
    softplus: bool = True

    @nn.compact
    def __call__(self, h1, d_ei): 
        assert d_ei.ndim <= 3
        d_ei = jnp.atleast_3d(d_ei)[:, :, -1:, None] # [n_elec, n_nucl, 1, 1]
        n_nucl = d_ei.shape[1]
        n_out = self.out_size * self.determinants
        pshape = (n_nucl, self.out_size, self.determinants)
        kernel_init = nn.initializers.variance_scaling(
            0.01, 'fan_in', 'truncated_normal')
        sigma = nn.Dense(
            n_out, 
            param_dtype=_t_real,
            kernel_init=kernel_init,
            bias_init=nn.initializers.ones
        )(h1[:n_nucl]).reshape(pshape)
        pi = nn.Dense(
            n_out, 
            param_dtype=_t_real,
            kernel_init=kernel_init,
            bias_init=nn.initializers.ones
        )(h1[:n_nucl]).reshape(pshape)
        if self.softplus:
            sigma = nn.softplus(sigma)
            pi = nn.softplus(pi)
        return jnp.sum(pi * jnp.exp(-sigma * d_ei), axis=1)
    

class OrbitalMap(nn.Module):
    spins: Tuple[int, int]
    determinants: int
    full_det: bool = True
    share_weights: bool = False

    @nn.compact
    def __call__(self, h1, d_ei):
        # h_one is [n_nucl+n_elec, n_desc]
        # d_ei is [n_elec, n_nucl, 1]
        n_up, n_dn = self.spins
        n_el, n_nu = d_ei.shape[:-1]
        assert n_up + n_dn == n_el
        n_det = self.determinants

        # make orbital from h1 and envelope
        def orbital_fn(h1_el, h1_nu, d_ei, n_orb):
            n_param = n_orb * n_det
            dense = nn.Dense(n_param, param_dtype=_t_real)
            envelope = IsotropicEnvelope(n_orb, self.determinants)
            assert envelope.out_size == n_orb
            # Actual orbital function
            return dense(h1_el).reshape(-1, n_orb, n_det) * envelope(h1_nu, d_ei) 

        # Case destinction for weight sharing 
        h1_nucl, h1_elec = jnp.split(h1, [n_nu])

        if self.share_weights:
            uu, dd = jnp.split(orbital_fn(h1_elec, h1_nucl, d_ei, max(self.spins)), 
                               self.spins[:1], axis=0)
            if self.full_det:
                ud, du = jnp.split(orbital_fn(h1_elec, h1_nucl, d_ei, max(self.spins)), 
                                   self.spins[:1], axis=0)
                orbitals = (jnp.concatenate([
                    jnp.concatenate([uu[:, :self.spins[0]], ud[:, :self.spins[1]]], axis=1),
                    jnp.concatenate([du[:, :self.spins[0]], dd[:, :self.spins[1]]], axis=1),
                ], axis=0),)
            else:
                orbitals = (uu[:, :self.spins[0]], dd[:, :self.spins[1]])
        else:
            h_by_spin = jnp.split(h1_elec, self.spins[:1], axis=0)
            d_by_spin = jnp.split(d_ei, self.spins[:1], axis=0)
            orbitals = tuple(
                orbital_fn(h, h1_nucl, d, no)
                for h, d, no in zip(
                    h_by_spin,
                    d_by_spin,
                    (sum(self.spins),)*2 if self.full_det else self.spins)
                )
            if self.full_det:
                orbitals = (jnp.concatenate(orbitals, axis=0),)
        return tuple(o.transpose(2, 0, 1) for o in orbitals)


class ElectronCusp(nn.Module):
    spins: tuple[int, int]
    
    @nn.compact
    def __call__(self, d_ee: Array) -> Array:
        w_para, w_anti = self.param('weight', fix_init, [1e-2]*2)
        a_para, a_anti = self.param('alpha', fix_init, [1.]*2)
        d_ee = jnp.atleast_3d(d_ee)[..., -1]
        uu, ud, du, dd = [
            s
            for split in jnp.split(d_ee, self.spins[:1], axis=0)
            for s in jnp.split(split, self.spins[:1], axis=1)
        ]
        same = jnp.concatenate([uu.reshape(-1), dd.reshape(-1)])
        diff = jnp.concatenate([ud.reshape(-1), du.reshape(-1)])
        result = -(1/4) * w_para * (a_para**2 / (a_para + same)).sum()
        result += -(1/2) * w_anti * (a_anti**2 / (a_anti + diff)).sum()
        return result


class FermiNet(FullWfn):
    elems: Sequence[int]
    spins: tuple[int, int]
    hidden_dims: Sequence[Tuple[int, int]] = ((64, 16),)*4
    determinants: int = 16
    full_det: bool = True
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

        h1, h2, dmat = raw_features(r, x)
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

        orbital_map = OrbitalMap((n_up, n_dn), self.determinants, 
                                 self.full_det, self.spin_symmetry)
        # tuple of [n_det, n_orb, n_orb]
        orbitals = orbital_map(h1, d_ei=dmat[n_nucl:, :n_nucl])

        signs, logdets = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0),
            *jax.tree_map(jnp.linalg.slogdet, orbitals)) # [1 or 2, n_det]
        signs, logdets = signs.prod(0), logdets.sum(0)
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
