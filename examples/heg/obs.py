#!/usr/bin/env python
import os
import numpy as np
import jax.numpy as jnp
from fvmc import observable as obs
from fvmc.wavefunction.heg import heg_rs
from fvmc.ewaldsum import gen_pbc_disp_fn

def rwsc(axes, dn=2):
    """ radius of the inscribed sphere inside the real-space
    Wigner-Seitz cell of the given cell

    Args:
      axes (np.array): lattice vectors in row-major
      dn (int,optional): number of image cells to search in each
       dimension. dn=1 searches 26 images in 3D.
    Returns:
      float: Wigner-Seitz cell radius
    """
    ndim = len(axes)
    from itertools import product
    r2imgl  = []  # keep a list of distance^2 to all neighboring images
    images = product(range(-dn, dn+1), repeat=ndim)
    for ushift in images:
        if sum(ushift) == 0:
            continue  # ignore self
        shift = np.dot(ushift, axes)
        r2imgl.append(np.dot(shift, shift))
    rimg = np.sqrt(min(r2imgl))
    return rimg/2.

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('ftraj', type=str)
    parser.add_argument('--fyml', type=str, default='hparams.yaml')
    parser.add_argument('--iter', '-i', type=int, default=0)
    parser.add_argument('--jter', '-j', type=int, default=None)
    parser.add_argument('--nx', '-n', type=int, default=48)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--spinor', '-s', action='store_true')
    args = parser.parse_args()

    # set up folder to cache results
    tname = os.path.basename(args.ftraj)
    cache_dir = 'cache-' + tname[:tname.rfind('.')] + '-obs'

    # read metadata
    meta_sys = obs.read_meta(args.fyml)
    cell = jnp.asarray(meta_sys['system']['cell'])
    nelec = meta_sys['system']['nelec']
    if args.spinor:
        spins = (nelec,)
    else:
        spins = meta_sys['system']['spins']
    nspin = len(spins)
    print('spins = ', spins)
    meta_sys = dict(spins=list(spins), cell=cell.tolist())

    # derived metadata
    ndim = len(cell)

    # read trajectories
    traj = obs.read_traj(args.ftraj, ndim, nelec)[args.iter:args.jter]
    niter, nwalker, nelec, ndim = traj.shape
    if niter < 4:  # !!!! HACK: split walkers for errorbar
        niter = 4
        traj = traj.reshape(niter, -1, nelec, ndim)

    # calculate observables

    nx = args.nx
    #   density rho(r)
    bins = (nx,)*ndim
    calc_dens = obs.gen_calc_dens(cell, spins, bins)
    meta_dens, rhoms, rhoes = obs.calc_obs(traj, calc_dens)
    #   save with processed metadata
    edges = meta_dens['edges']
    meta = dict(aname='density')
    meta.update(meta_sys)
    for i, e in enumerate(edges):
        meta['edge%d' % i] = e.tolist()
    obs.save_obs('%s/dens' % cache_dir, meta, rhoms, rhoes)

    #   pair correlation g(r)
    #   isotropic
    rcut = rwsc(cell)
    calc_gofr = obs.gen_calc_gofr(cell, spins, nx, rcut)
    meta_gofr, grms, gres = obs.calc_obs(traj, calc_gofr)
    #   save with processed metadata
    r = meta_gofr['r']
    meta = dict(aname='gofr', r=r.tolist())
    meta.update(meta_sys)
    obs.save_obs('%s/gofr' % cache_dir, meta, grms, gres)

    #   vector
    calc_pair_hist = obs.gen_calc_pair_hist(cell, spins, bins)
    meta_gv, gvms, gves = obs.calc_obs(traj, calc_pair_hist)
    #   save with processed metadata
    meta = dict(aname='vecgofr')
    for i, e in enumerate(meta_gv['edges']):
      meta['edge%d' % i] = e.tolist()
    meta.update(meta_sys)
    obs.save_obs('%s/vecgofr' % cache_dir, meta, gvms, gves)

    #   S(k)
    rs = heg_rs(cell, nelec)
    kcut = 6./rs
    calc_sofk = obs.gen_calc_sofk(cell, spins, kcut)
    meta_sofk, skams, skaes = obs.calc_obs(traj, calc_sofk)
    rkms = skams[:nspin]
    rkes = skaes[:nspin]
    skms = skams[nspin:].real
    skes = skaes[nspin:].real
    #   save with processed metadata
    kvecs = meta_sofk['kvecs'].tolist()
    meta = dict(aname='rhok', kvecs=kvecs)
    meta.update(meta_sys)
    obs.save_obs('%s/rhok' % cache_dir, meta, rkms, rkes)
    meta = dict(aname='sofk', kvecs=kvecs)
    meta.update(meta_sys)
    obs.save_obs('%s/sofk' % cache_dir, meta, skms, skes)

    # extra processing for fluctuating S(k)
    dskl = []
    k = 0
    for ii, rkmi in enumerate(rkms):
        for _, rkmj in enumerate(rkms[ii:], start=ii):
            dskm = skms[k] - (rkmi*rkmj.conj())
            dskl.append(dskm.real)
            k += 1
    meta = dict(aname='dsk', kvecs=kvecs)
    meta.update(meta_sys)
    obs.save_obs('%s/dsk' % cache_dir, meta, dskl, skes)

if __name__ == '__main__':
  main()
