# MIT License
#
# Copyright (c) 2019 Lucas K Wagner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.

# Further modification has been done by Yixiao Chen

import jax
import jax.numpy as jnp

# from . import LOGGER
from .utils import displace_matrix, split_spin


class EwaldSum:
    """
    Using the Ewald summation to calculate the Coulomb potential

    Unlike original version, this simplified one treats
    nuclei and electrons on the equal footing and calculate at once.
    """

    def __init__(self, latvec, g_max=200, n_lat=1, g_threshold=1e-12, alpha=None):
        """
        Initilization of the Ewald summation class by preparing
        pbc displace function, lattice displacements, and reciporcal g points

        Args:
            latvec (Array): 3x3 matrix with each row a lattice vector
            g_max (int): How far to take reciprocal sum; probably never needs to be changed.
            n_lat (int): How far to take real-space sum; probably never needs to be changed.
            g_threshold (float): ignore g points below this value. Following DeepSolid value.
        """
        self.latvec = jnp.asarray(latvec)
        self.recvec = jnp.linalg.inv(latvec).T
        self.cellvolume = jnp.abs(jnp.linalg.det(latvec))
        # determine alpha
        smallest_height = jnp.min(1 / jnp.linalg.norm(self.recvec, axis=1))
        self.alpha = 5.0 / smallest_height if alpha is None else alpha
        # minimal image displacement function
        self.disp_fn = gen_pbc_disp_fn(latvec)
        # lattice displacement to be added to disp in real space sum
        self.lattice_displacements = gen_lattice_displacements(latvec, n_lat)
        lat_norm = jnp.linalg.norm(self.lattice_displacements, axis=-1)
        lat_norm = lat_norm[lat_norm > 0]
        self.simg_const = jnp.sum(jax.lax.erfc(self.alpha * lat_norm) / lat_norm)
        # genetate g points to be used in reciprocal sum. Keep only large gweights
        raw_gpoints = gen_positive_gpoints(self.recvec, g_max)
        raw_gweight = calc_gweight(raw_gpoints, self.cellvolume, self.alpha)
        selected_gidx = raw_gweight > g_threshold
        self.gpoints = raw_gpoints[selected_gidx]
        self.gweight = raw_gweight[selected_gidx]

    def const_part(self, charge):
        dm1 = self.latvec.shape[-1] - 1
        q_sum = jnp.sum(charge)
        q2_sum = jnp.sum(charge ** 2)
        e_self = - self.alpha / jnp.sqrt(jnp.pi) * q2_sum
        denom = dm1 * self.cellvolume * self.alpha ** dm1
        e_charged = - jnp.pi ** (dm1/2.) / denom * q_sum ** 2
        return e_self, e_charged

    def real_part(self, charge, pos):
        if charge.shape[0] < 2:
            return 0
        disp = displace_matrix(pos, pos, disp_fn=self.disp_fn)
        rvec = disp[None, :, :, :] + self.lattice_displacements[:, None, None, :]
        r = jnp.linalg.norm(rvec + jnp.eye(pos.shape[0])[..., None], axis=-1)
        charge_ij = charge[:, None] * charge[None, :]
        e_real = jnp.sum(jnp.triu(charge_ij * jax.lax.erfc(self.alpha * r) / r, k=1))
        e_real += 0.5 * jnp.sum(charge ** 2) * self.simg_const # self image
        return e_real

    def recip_part(self, charge, pos):
        g_dot_r = self.gpoints @ pos.T # [n_gpoints, n_particle]
        sfactor = jnp.exp(1j * g_dot_r) @ charge # [n_gpoints,]
        e_recip = self.gweight @ (sfactor * sfactor.conj())
        return e_recip.real

    def energy(self, charge, pos):
        """Calculation the Coulomb energy from point charges and their positions"""
        return (sum(self.const_part(charge))
                + self.real_part(charge, pos)
                + self.recip_part(charge, pos))

    def calc_pe(self, elems, r, x):
        """Warpped interface for potential energy from nuclei and electrons"""
        assert elems.shape[0] == r.shape[0]
        x, _ = split_spin(x) # Coulomb is spin independent
        assert elems.ndim == 1 and r.ndim == x.ndim == 2
        charge = jnp.concatenate([elems, -jnp.ones(x.shape[0])], axis=0)
        pos = jnp.concatenate([r, x], axis=0)
        return self.energy(charge, pos)


def determine_cell_type(latvec, ortho_tol=1e-10) -> str:
    is_diagonal = jnp.all(jnp.abs(latvec - jnp.diag(jnp.diag(latvec))) < ortho_tol)
    if is_diagonal:
        return "diagonal"
    is_orthogonal = jnp.all(jnp.abs(jnp.triu(latvec @ latvec.T, k=1)) < ortho_tol)
    if is_orthogonal:
        return "orthogonal"
    return "general"


def gen_pbc_disp_fn(latvec, mode="auto"):
    latvec = jnp.asarray(latvec)
    mode = mode.lower()
    if mode == "auto":
        ortho_tol = 1e-10
        mode = determine_cell_type(latvec, ortho_tol=ortho_tol)
    # diagonal cell
    if mode.startswith("diag"):
        latdiag = jnp.diagonal(latvec)
        def diagonal_disp(xa, xb):
            disp = xa - xb
            frac_disp = disp/latdiag
            shifted_frac_disp = frac_disp - jnp.rint(frac_disp)
            return shifted_frac_disp * latdiag
        return diagonal_disp
    # orthogonal cell
    if mode.startswith("orth"):
        invvec = jnp.linalg.inv(latvec)
        def orthogonal_disp(xa, xb):
            disp = xa - xb
            frac_disp = disp @ invvec
            shifted_frac_disp = frac_disp - jnp.rint(frac_disp)
            return shifted_frac_disp @ latvec
        return orthogonal_disp
    # general cell
    if mode.startswith("gen"):
        n_lat = 1
        images = gen_lattice_displacements(latvec, n_lat)
        invvec = jnp.linalg.inv(latvec)
        def xpbc(x):  # wrap position into simulation cell
            f = x @ invvec
            return (f % 1) @ latvec
        def monoclinic_disp(xa, xb):
            disps = (xpbc(xa) - xpbc(xb))[None] + images
            dists = jnp.linalg.norm(disps, axis=-1)
            idx = jnp.argmin(dists)
            return disps[idx]
        return monoclinic_disp
    # fail to recognize mode
    raise ValueError(f"unknown mode for gen_pbc_disp_fn: {mode}")


def gen_lattice_displacements(latvec, n_lat):
    n_d = latvec.shape[0] # number of spacial dimension
    XYZ = jnp.meshgrid(*[jnp.arange(-n_lat, n_lat + 1)] * n_d, indexing="ij")
    xyz = jnp.stack(XYZ, axis=-1).reshape((-1, n_d))
    return jnp.asarray(jnp.dot(xyz, latvec))


def gen_positive_gpoints(recvec, g_max):
    # Determine G points to include in reciprocal Ewald sum
    n_d = recvec.shape[0] # number of spacial dimension
    zero = jnp.asarray([0])
    half = jnp.arange(1, g_max + 1)
    full = jnp.arange(-g_max, g_max + 1)
    gpts_list = [jnp.meshgrid(
        *([zero]*ii + [half] + [full]*(n_d-ii-1)),
        indexing='ij'
    ) for ii in range(n_d)]
    gpts = jnp.concatenate([
        jnp.stack(g, axis=-1).reshape(-1, n_d)
        for g in gpts_list
    ], axis=0)
    gpoints = 2 * jnp.pi * gpts @ recvec
    return gpoints


def calc_gweight(gpoints, cellvolume, alpha):
    if gpoints.shape[-1] == 2:
        gnorm = jnp.linalg.norm(gpoints, axis=-1)
        gweight = (2 * jnp.pi / (cellvolume * gnorm)
                   * jax.lax.erfc(gnorm / ( 2 * alpha)))
    else: # 3d case
        gsquared = (gpoints ** 2).sum(-1)
        gweight = (4 * jnp.pi / (cellvolume * gsquared)
                   * jnp.exp(-gsquared / (4 * alpha ** 2)))
    return gweight
