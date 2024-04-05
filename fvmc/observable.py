"""Analyze a trajectory of walkers for observables.

traj shape = (niter, nsample, nelec, ndim)
walker shape = (nsample, nelec, ndim)
snapshot shape = (nelec, ndim)

gen_calc_[obs] are often needed to jit calc_[obs].
calc_[obs] typically operates on walker, and sometimes works for snapshot.

Typical usage:
>>> meta = read_meta('hparam.yml'); spins = meta['system']['spins']
>>> cell = meta['system']['cell']; ndim = len(cell)
>>> nelec = meta['system']['nelec']
>>> traj = read_traj('trajdump.npy', ndim, nelec)
>>> bins = (64,)*ndim  # density grid
>>> rho_info, rho_mean, rho_error = calc_obs(traj, calc_dens, cell, spins, bins)
>>> # calc_dens is directly jittable, so no gen_calc_dens and pass args
>>> rcut = cell[0, 0]/2  # g(r) radial cutoff
>>> calc_gofr = gen_calc_gofr(cell, spins, bins[0], rcut)
>>> # calc_gofr is jitted during gen_calc_gofr with args baked in
>>> gr_info, gr_mean, gr_error = calc_obs(traj, calc_gofr)
"""

from functools import partial
from typing import Callable, Tuple, Literal

import jax
import numpy as np
import yaml
from jax import lax
from jax import numpy as jnp

from .ewaldsum import gen_pbc_disp_fn, gen_positive_gpoints
from .wavefunction.base import log_psi_from_model
from .utils import Array, displace_matrix, parse_spin_num, pdist
from .utils import ArrayTree, gen_ksphere
from .utils import ElecConf, ensure_spin


def read_meta(fmeta: str) -> dict:
    """ read metadata of run

    Args:
      fmeta (str): hparams.yaml
    Return:
      dict: metadata
    """
    # read raw metadata
    with open(fmeta, 'r') as f:
      meta = yaml.full_load(f)
    # derived metadata
    elems = meta['system']['elems']
    # nelec
    nelec = sum(elems) if elems else 0
    nelec -= meta['system']['charge']
    meta['system']['nelec'] = nelec
    # spins
    spin = meta['system']['spin']
    spins = parse_spin_num(nelec, spin)
    # !!!! HACK add following to parse_spin?
    for i, ns in enumerate(spins):
      if ns <= 0:
        spins = spins[:i]
        break
    meta['system']['spins'] = spins
    return meta


def read_traj(ftraj: str, ndim: int, nelec: int) -> ElecConf:
    """Read trajectory dump into array

    Args:
      ftraj (str): trajdump.npy
      ndim (int): number of spatial dimensions
      nelec (int): number of electrions
    Return:
      Array: particle coordinates, shape (niter, nsample, nelec, ndim)
      Array: spin coordinates, shape (niter, nsample, nelec)
    """
    data = np.load(ftraj)
    niter, nsample, ncoord = data.shape
    ned = nelec * ndim
    # HACK assume the spin variable is at tail here
    traj = data[:, :, :ned].reshape(niter, nsample, nelec, ndim)
    straj = data[:, :, ned:]
    return traj, straj


def reshape_traj(traj: Array, batch_size:int, max_batch:int=None):
    """Reshape trajectory to limit memory and time costs

    Args:
      traj (Array): trajectory with shape (niter, nsample, **nconf)
      batch_size (int): number of configs. in a batch, limit memory cost
      max_batch (int, optional): maximum number of batches, limit compute
    Return:
      Array: selected subset of given trajectory
    """
    traj_shape = traj.shape
    niter, nsample = traj_shape[:2]
    conf_shape = traj_shape[2:]
    ntot = niter * nsample
    nbatch = ntot // batch_size
    if nbatch < 4:
        msg = 'Too few batches %d for statistical error.' % nbatch
        msg += '  reduce batch_size to ntot//4=%d (ntot=%d).' % (ntot//4, ntot)
        raise RuntimeError(msg)
    max_batch = min(max_batch, nbatch) if max_batch else nbatch
    nleft = ntot % batch_size
    nevery = nbatch // max_batch
    flat_traj = traj.reshape(-1, *conf_shape)[nleft:]
    return flat_traj.reshape(-1, batch_size, *conf_shape)[::nevery][:max_batch]


def calc_obs(traj: ElecConf, calc_func: Callable, *args, **kwargs):
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
        y2 = jnp.mean(y1 * y1, axis=0)
        # !!!! HACK: assume no autocorrelation
        ye = (y2 - ym * ym)**0.5 / nsamp**0.5
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
    # normalized so that the integral of rho is the number of particles
    # numerically, sum(hist * bin_vol) = n_elec
    nsample, nelec, ndim = walker.shape
    hrange = [(0, 1)] * ndim
    bin_vol = np.prod(np.diff(hrange, axis=-1)) / np.prod(bins)
    split_idx = np.cumsum((spins))[:-1]
    invvec = jnp.linalg.inv(cell)
    walker_split = jnp.split(walker, split_idx, axis=-2)
    res = []
    for ws in walker_split:
        fracs = jnp.dot(ws.reshape(-1, ndim), invvec) % 1
        hist, edges = jnp.histogramdd(fracs, bins=bins, range=hrange)
        res.append(hist / nsample / bin_vol)
    return res, dict(edges=edges)


# =============================== sofk ==============================

def select_kvecs(recvec: Array, kcut: float, g_max: int=200) -> Array:
    kvecs = gen_positive_gpoints(recvec, g_max)
    k2 = jnp.sum(kvecs * kvecs, axis=-1)
    sel = k2 < (kcut * kcut)
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
            rhok = jnp.sum(jnp.exp(1j * kr), axis=-1)
            rhoks.append(rhok)
            rkm = rhok.mean(axis=-1)
            res.append(rkm)
        for ii, rhoki in enumerate(rhoks):
            for _, rhokj in enumerate(rhoks[ii:], start=ii):
                sofk = (rhoki * rhokj.conj()).real # real means symmetrize ij
                res.append(sofk.mean(axis=-1))
        return res, dict(kvecs=kvecs)

    return calc_sofk


# =============================== gofr ==============================

def gen_calc_vecgofr(
        cell: Array, spins: Tuple[int], bins: Tuple[int],
        normalize: Literal["spin", "charge"] = "spin"
) -> Callable:
    # normalized so that for uncorrelated particles this is one
    # `normalize = 'spin'` means each spin species is normalized separately
    # `normalize = 'charge'` means all particles are normalized together
    nelec = sum(spins)
    disp_fn = gen_pbc_disp_fn(cell)
    invvec = jnp.linalg.inv(cell)
    hlim = (-0.5, 0.5)
    bin_vol = np.prod(np.diff(hlim, axis=-1) / bins)
    norms = []
    for ii, ni in enumerate(spins):
        for jj, nj in enumerate(spins[ii:], start=ii):
            npair = {"spin": ni * (ni - 1) if ii == jj else 2 * ni * nj,
                     "charge": nelec * (nelec - 1)}[normalize]
            norms.append(1 / (bin_vol * npair))

    @jax.jit
    def calc_vecgofr(walker: Array):
        disp = jax.vmap(partial(displace_matrix, disp_fn=disp_fn))(walker, walker)
        disp = disp @ invvec
        nsample = len(disp)
        hists, edges = histogram_spin_blocks(disp, bins, spins, (-0.5, 0.5))
        res = [h * n / nsample for h, n in zip(hists, norms)]
        return res, dict(edges=edges)

    return calc_vecgofr


def gen_calc_gofr(
        cell: Array, spins: Tuple[int],
        bins: int, rcut: float,
        normalize: Literal["spin", "charge"] = "spin"
) -> Callable:
    # normalized so that for uncorrelated particles this is one
    # `normalize = 'spin'` means each spin species is normalized separately
    # `normalize = 'charge'` means all particles are normalized together
    nelec = sum(spins)
    edges = np.histogram_bin_edges([0], bins, (0, rcut))
    disp_fn = gen_pbc_disp_fn(cell)
    gr_factor = gofr_norm_factor(cell, edges)
    norms = []
    for ii, ni in enumerate(spins):
        for jj, nj in enumerate(spins[ii:], start=ii):
            npair = {"spin": ni * (ni - 1) if ii == jj else 2 * ni * nj,
                     "charge": nelec * (nelec - 1)}[normalize]
            norms.append(gr_factor / npair)

    @jax.jit
    def calc_gofr(walker: Array):
        dist = jax.vmap(partial(pdist, disp_fn=disp_fn))(walker)  # [-1, nelec, nelec]
        dist = dist[:, :, :, None] # [-1, nelec, nelec, ndim]
        nsample = dist.shape[0]
        # count
        hists, edges = histogram_spin_blocks(dist, bins, spins, (0, rcut))
        # grid
        e = jnp.asarray(edges[0])
        r = 0.5 * (e[1:] + e[:-1])
        # normalize
        res = []
        for hist, norm in zip(hists, norms):
            gr = hist.at[0].set(0) * norm / nsample
            res.append(gr)
        return res, dict(r=r)

    return calc_gofr


def gofr_norm_factor(cell: Array, bin_edges: Array) -> Array:
    r"""V / (4 \pi r^2 dr) for 3D, V / (2 \pi r dr) for 2D"""
    ndim, ndim = cell.shape
    # calculate shell volume of bins
    vnorm = np.diff(2 * (ndim-1) / ndim * np.pi * bin_edges**ndim)
    # total cell volume V
    volume = np.abs(np.linalg.det(cell))
    return volume / vnorm


def histogram_spin_blocks(
        dis: Array, bins: Tuple[int],
        spins: Tuple[int], xlim: Tuple[float]
) -> Tuple[list, Array]:
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
                dflat = dblock[:, mask].reshape(-1, ndim)
            else:
                dblock_t = dis_block[jj][ii]
                dflat = jnp.concatenate([dblock.reshape(-1, ndim),
                                         dblock_t.reshape(-1, ndim)], axis=0)
            hist, edges = jnp.histogramdd(dflat, bins=bins, range=[xlim] * ndim)
            res.append(hist)
    return res, edges


# =============================== nofk ==============================

def gen_calc_nofk(
        cell: Array, kcut: float, twist: Array, n_elec: int,
        ansatz, params: ArrayTree, key: Array, n_samp: int = 32,
) -> Callable:
    n_dim = cell.shape[-1]
    # mask to make new snapshots, each with one electron moved
    single_particle_mask = jnp.zeros([n_elec, n_elec, n_dim], dtype=bool)
    for i in range(n_elec):
        single_particle_mask = single_particle_mask.at[i, i, :].set(True)
    # prepare to calculate wf ratios
    batched_logpsi = jax.vmap(log_psi_from_model(ansatz), in_axes=[None, 0])
    # locations to evaluate n(k) on
    kvecs = gen_ksphere(cell, kcut, twist=twist)
    # quadraqure points for Fourier transform
    key, subkey = jax.random.split(key)
    pos1 = jax.random.uniform(subkey, (n_samp, n_dim)) @ cell

    def calc_nofk_snapshot_one_particle(pos: Array, p1: Array, logpsi0: float) -> Array:
        dr = p1 - pos  # move vectors, one for each particle [n_e, n_dim]
        eikr = jnp.exp(1j * jnp.tensordot(kvecs, -dr, (-1, -1)))  # [n_k, n_e]
        # move each particle to p1
        dpos = dr[None] * single_particle_mask  # [n_e, n_e, n_dim]
        r1 = pos[None] + dpos
        # evaluate wf ratios (all particles to one location)
        logpsi1 = batched_logpsi(params, r1)
        ratio = jnp.exp(logpsi1 - logpsi0)[None]  # same ratio for all k [-1, n_e]
        nk1 = (eikr.real * ratio.real - eikr.imag * ratio.imag).sum(axis=-1)
        return nk1

    def calc_nofk_snapshot(pos: Array) -> Array:
        logpsi0 = batched_logpsi(params, pos[None])[0]
        one_fn = jax.vmap(calc_nofk_snapshot_one_particle, in_axes=(None, 0, None))
        nk = one_fn(pos, pos1, logpsi0).mean(axis=0)
        return nk

    @jax.jit
    def calc_nofk(walker: Array):
        nk = lax.map(calc_nofk_snapshot, walker).mean(axis=0)
        nk_spins = [nk]  # TODO: spin-resolved n(k)
        return nk_spins, dict(kvecs=kvecs)

    return calc_nofk


# =============================== mdens ==============================

def pauli_ss(s0, s1):
    r"""Project continuous spin to Pauli matrices <s0|\sigma|s1>"""
    sx = 2 * jnp.cos(s0 + s1)
    sy = 2 * jnp.sin(s0 + s1)
    sz = 2j * jnp.sin(s0 - s1)
    return sx, sy, sz


# this is the slow backup plan if no `pauli` method available
def project_spins(ansatz, params, x: ElecConf) -> Array:
    # prepare to calculate wf ratios
    logpsi_fn = log_psi_from_model(ansatz)
    batched_logpsi = jax.vmap(logpsi_fn, in_axes=[None, 0])
    # simple integration
    int1d = jax.scipy.integrate.trapezoid
    # exact spin integration by Simpson's rule
    ng = 9
    sgrid = jnp.linspace(0, 2*jnp.pi, ng)
    # s on the left, wfn at the denominator
    logpsi0 = logpsi_fn(params, x)
    r0, s0 = ensure_spin(x)
    nelec, ndim = r0.shape
    # inner loop for s1 on the right side, <s|\sigma|s1> psi(s1) / psi(s)
    def evaluate_spin_projections(s1: float):
        """At a given spin value, evaluate projection of all electrons
        onto Pauli matrices.

        Args:
            s1 (float): spin value on integration grid
        Return:
            Array: grid values with shape (nelec, 3)
        """
        news = s1 * jnp.eye(nelec) + s0 * (1-jnp.eye(nelec))
        newx = (jnp.broadcast_to(r0, (nelec, nelec, ndim)), news)
        logpsi1 = batched_logpsi(params, newx)
        ratio = jnp.exp(logpsi1 - logpsi0) # psi1 on top, psi0 on bottom
        # s1 to be integrated is on the right
        return jnp.stack(pauli_ss(s0, s1), axis=-1) * ratio[..., None]
    # get spin integration grids (ns, nelec, 3) for (nelec e, 3 Paulis)
    grids = jax.lax.map(evaluate_spin_projections, sgrid)
    # integrate s1
    weights = int1d(grids, x=sgrid, axis=0) / (2*jnp.pi)
    return weights


def gen_calc_mdens(
        cell: Array, bins: Tuple[int],
        ansatz, params: ArrayTree
) -> Callable:
    # same normalization as density
    hrange = [(0, 1)] * len(bins)
    bin_vol = np.prod(np.diff(hrange, axis=-1)) / np.prod(bins)
    # prepare to calculate fractional coordinates
    invvec = jnp.linalg.inv(cell)
    if hasattr(ansatz, 'pauli'):
        batched_pauli = jax.vmap(partial(ansatz.apply, params, method='pauli'))
    else:
        from . import LOGGER
        LOGGER.warning('No `pauli` method available, using backup integration method. '
                       'This will be VERY SLOW.')
        batched_pauli = lambda x: lax.map(partial(project_spins, ansatz, params), x)

    @ jax.jit
    def calc_mdens(walker: ElecConf):
        r, s = ensure_spin(walker)
        nsample, nelec, ndim = r.shape
        fracs = (r.reshape(-1, ndim) @ invvec) % 1
        weights = batched_pauli(walker).real # (nsample, nelec, 3)
        wx, wy, wz = weights.reshape(-1, 3).T # (3, nsample * nelec)
        # histogram
        hx, edges = jnp.histogramdd(fracs, bins=bins, range=hrange, weights=wx)
        hy, edges = jnp.histogramdd(fracs, bins=bins, range=hrange, weights=wy)
        hz, edges = jnp.histogramdd(fracs, bins=bins, range=hrange, weights=wz)
        mags = jnp.stack([hx, hy, hz], axis=0) / nsample / bin_vol
        meta = {'edges': edges}
        return [mags], meta

    return calc_mdens


# =============================== utils ==============================

def default_argparse(max_batch=64):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('ftraj', type=str)
    parser.add_argument('--fyml', type=str, default='hparams.yaml')
    parser.add_argument('--iter', '-i', type=int, default=0)
    parser.add_argument('--jter', '-j', type=int, default=None)
    parser.add_argument('--max_batch', '-m', type=int, default=max_batch,
        help='reduce to save time')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
        help='reduce to save memory')
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser
