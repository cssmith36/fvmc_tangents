"""Analyze a trajectory of walkers for observables.

traj shape = (niter, nsample, nelec, ndim)
walker shape = (nsample, nelec, ndim)
snapshot shape = (nelec, ndim)

gen_calc_[obs] are often needed to jit calc_[obs].
calc_[obs] typically operates on walker, and sometimes works for snapshot.

Typical usage:
>>> meta = read_meta('hparam.yml'); spins = meta['system']['spins']
>>> cell = meta['system']['cell']; ndim = len(cell)
>>> traj = read_traj('trajdump.npy')
>>> bins = (64,)*ndim  # density grid
>>> rho_info, rho_mean, rho_error = calc_obs(traj, calc_dens, cell, spins, bins)
>>> # calc_dens is directly jittable, so no gen_calc_dens and pass args
>>> rcut = cell[0, 0]/2  # g(r) radial cutoff
>>> calc_gofr = gen_calc_gofr(cell, spins, bins[0], rcut)
>>> # calc_gofr is jitted during gen_calc_gofr with args baked in
>>> gr_info, gr_mean, gr_error = calc_obs(traj, calc_gofr)
"""

from functools import partial
from typing import Callable, Tuple

import jax
import numpy as np
import yaml
from jax import lax
from jax import numpy as jnp

from .ewaldsum import gen_pbc_disp_fn, gen_positive_gpoints
from .utils import Array, displace_matrix, parse_spin_num, pdist


def read_meta(fmeta: str) -> dict:
    """ read metadata of run

    Args:
      fmeta (str): hparams.yaml
    Return:
      dict: metadata
    """
    # read raw metadata
    with open(fmeta, 'r') as f:
      meta = yaml.safe_load(f)
    # derived metadata
    #   nelec
    #   !!!! HACK only works for heg
    nelec = -meta['system']['charge']
    meta['system']['nelec'] = nelec
    #   spins
    spin = meta['system']['spin']
    spins = parse_spin_num(nelec, spin)
    #   !!!! HACK add following to parse_spin?
    for i, ns in enumerate(spins):
      if ns <= 0:
        spins = spins[:i]
        break
    meta['system']['spins'] = spins
    return meta


def read_traj(ftraj : str, ndim : int) -> Array:
    """ read trajectory dump into array

    Args:
      ftraj (str): trajdump.npy
      ndim (int): number of spatial dimensions
    Return:
      Array: particle coordinates, shape (niter, nsample, nelec, ndim)
    """
    data = np.load(ftraj)
    niter, nsample, nelec_ndim = data.shape
    nelec = nelec_ndim//ndim
    if nelec*ndim != nelec_ndim:
        msg = ('read_traj failed to infer dimensions: '
              f'Ne={nelec} ndim={ndim} but Ne*ndim={nelec_ndim}')
        raise RuntimeError(msg)
    traj = data.reshape(niter, nsample, nelec, ndim)
    return traj


def calc_obs(traj: Array, calc_func: Callable, *args, **kwargs):
    """Calculate observable from trajectory.

    Each calc_[obs](W_i, ...) computes observable O_i for walker W_i,
      and returns (list, dict) containing (data, meta).
    Given i=1,niter O_i, errorbar is estimated as sqrt( <O_i^2> - <O_i>^2 )/sqrt(niter).

    Args:
      traj (Array): trajectory
      calc_func (Callable): function to calculate observable from walker
    Return:
      (dict, Array, Array): (meta, mean, error)
    """
    partial_calc = lambda walker: calc_func(walker, *args, **kwargs)
    outputs, metas = lax.map(partial_calc, traj)
    meta = jax.tree_map(lambda x: x[0], metas)
    # calculate mean and error
    yml = []
    yel = []
    for y1 in outputs:
        nsamp = y1.shape[0]
        ym = jnp.mean(y1, axis=0)
        y2 = jnp.mean(y1*y1, axis=0)
        # !!!! HACK: assume no autocorrelation
        ye = (y2-ym*ym)**0.5/nsamp**0.5
        yml.append(ym)
        yel.append(ye)
    return meta, jnp.asarray(yml), jnp.asarray(yel)


def save_obs(prefix: str, meta: dict, ym: Array, ye: Array) -> None:
    import os
    path = os.path.dirname(prefix)
    if not os.path.isdir(path):
        os.makedirs(path)
    fyml = f'{prefix}-info.yml'
    with open(fyml, 'w') as f:
        yaml.dump(meta, f)
    for y, name in zip([ym, ye], ['mean', 'error']):
        fnpy = f'{prefix}-{name}.npy'
        np.save(fnpy, y)


def load_obs(prefix: str):
    fyml = f'{prefix}-info.yml'
    with open(fyml, 'r') as f:
        meta = yaml.safe_load(f)
    ret = [meta]
    for name in ['mean', 'error']:
        fnpy = f'{prefix}-{name}.npy'
        data = np.load(fnpy)
        ret.append(data)
    return ret


# =============================== dens ==============================

def gen_calc_dens(cell: Array, spins: Tuple[int], bins: Tuple[int]):
    return partial(calc_dens, cell=cell, spins=spins, bins=bins)


@partial(jax.jit, static_argnames=('bins', 'spins'))
def calc_dens(walker: Array, cell: Array, spins: Tuple[int], bins: Tuple[int]):
    nnr = np.prod(bins)
    nsample, nelec, ndim = walker.shape
    split_idx = np.cumsum((spins))[:-1]
    invvec = jnp.linalg.inv(cell)
    walker_split = jnp.split(walker, split_idx, axis=-2)
    res = []
    for ws in walker_split:
        fracs = jnp.dot(ws.reshape(-1, ndim), invvec) % 1
        hist, edges = jnp.histogramdd(fracs, bins=bins, range=[(0, 1)]*ndim)
        res.append(hist*nnr/nsample)
    return res, dict(edges=edges)


# =============================== sofk ==============================

def select_kvecs(recvec: Array, kcut: float, g_max: int=200) -> Array:
    kvecs = gen_positive_gpoints(recvec, g_max)
    k2 = jnp.sum(kvecs*kvecs, axis=-1)
    sel = k2 < kcut*kcut
    return kvecs[sel]


def gen_calc_sofk(cell: Array, spins: Tuple[int], kcut: float) -> Callable:
    recvec = jnp.linalg.inv(cell).T
    kvecs = select_kvecs(recvec, kcut)

    @jax.jit
    def calc_sofk(walker: Array):
        nspin = len(spins)
        spin_idx = np.cumsum(spins)[:-1]
        walker_split = jnp.split(walker, spin_idx, axis=-2)
        res = []
        rhoks = []
        for ii in range(nspin):
            kr = jnp.tensordot(kvecs, walker_split[ii], (-1, -1))
            rhok = jnp.sum(jnp.exp(1j*kr), axis=-1)
            rhoks.append(rhok)
            rkm = rhok.mean(axis=-1)
            res.append(rkm)
        for ii, rhoki in enumerate(rhoks):
            for _, rhokj in enumerate(rhoks[ii:], start=ii):
                sofk = (rhoki*rhokj.conj())  # symmetrize i j?
                res.append(sofk.mean(axis=-1))
        return res, dict(kvecs=kvecs)

    return calc_sofk


# =============================== gofr ==============================

def gen_calc_pair_hist(cell: Array, spins: Tuple[int], bins: Tuple[int]) -> Callable:
    disp_fn = gen_pbc_disp_fn(cell)
    invvec = jnp.linalg.inv(cell)

    @jax.jit
    def calc_pair_hist(walker: Array):
        disp = jax.vmap(partial(displace_matrix, disp_fn=disp_fn))(walker, walker)
        disp = disp @ invvec
        nsample = len(disp)
        hists, edges = histogram_spin_blocks(disp, bins, spins, (-0.5, 0.5))
        res = [h/nsample for h in hists]
        return res, dict(edges=edges)

    return calc_pair_hist


def gen_calc_gofr(cell: Array, spins: Tuple[int], bins: int, rcut: float) -> Callable:
    edges = np.arange(0, rcut+rcut/bins/2, rcut/bins)
    disp_fn = gen_pbc_disp_fn(cell)
    norms = []
    for ii, ni in enumerate(spins):
        for jj, nj in enumerate(spins[ii:], start=ii):
            n2 = None if ii == jj else nj
            gr_norm = gofr_norm(cell, edges, ni, n2=n2)
            norms.append(gr_norm/2)  # !!!! HACK: why /2?

    @jax.jit
    def calc_gofr(walker: Array):
        dist = jax.vmap(partial(pdist, disp_fn=disp_fn))(walker)  # [-1, nelec, nelec]
        dist = dist[:, :, :, None]/rcut # [-1, nelec, nelec, ndim]
        nsample = len(dist)
        # count
        hists, edges = histogram_spin_blocks(dist, bins, spins, (0, 1))
        # grid
        e = jnp.asarray(edges[0])*rcut
        r = 0.5*(e[1:]+e[:-1])
        # normalize
        res = []
        for hist, norm in zip(hists, norms):
          gr = hist.at[0].set(0)*norm/nsample
          res.append(gr)
        return res, dict(r=r)

    return calc_gofr


def gofr_norm(cell: Array, bin_edges: Array, n1: int, n2: int=None) -> Array:
    n2 = n2 if n2 else n1-1
    ndim, ndim = cell.shape
    # calculate volume of bins
    vnorm = np.diff(2*(ndim-1)/ndim*np.pi*bin_edges**ndim)
    # calculate density normalization
    npair = n1*n2/2
    volume = np.abs(np.linalg.det(cell))
    rho = npair/volume
    # assemble the norm vector
    gr_norm = 1./(rho*vnorm)
    return gr_norm


def histogram_spin_blocks(dis: list[list[Array]], bins: Tuple[int],
    spins: Tuple[int], xlim: Tuple[float]) -> Tuple[list, Array]:
    ndim = dis.shape[-1]
    # split displacement or distnaces by spins
    split_idx = np.cumsum((spins))[:-1]
    dis_split = jnp.split(dis, split_idx, axis=-2)
    dis_block = [jnp.split(d, split_idx, axis=-3) for d in dis_split]
    # histogram by spin-spin pair
    res = []
    for ii in range(len(spins)):
        for jj, nj in enumerate(spins[ii:], start=ii):
            dblock = dis_block[ii][jj]
            if ii == jj:
                with jax.ensure_compile_time_eval():
                    mask = ~jnp.eye(nj, dtype=bool)
                dblock = dblock[:, mask]
            dblock = dblock.reshape(-1, ndim)
            hist, edges = jnp.histogramdd(dblock, bins=bins, range=[xlim]*ndim)
            res.append(hist)
    return res, edges
