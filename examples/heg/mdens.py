#!/usr/bin/env python3
import os
import jax.numpy as jnp
from importlib import import_module
from typing import Callable
from fvmc import observable as obs
from fvmc.wavefunction.heg import heg_rs
from fvmc.utils import load_pickle

def import_function(path: str) -> Callable:
    """import function at run time

    Args:
      path (str): path to function, e.g., "numpy.linalg.norm"
    Return:
      Callable: imported function, e.g., norm
    """
    mod_path, fn_name = path.rsplit('.', maxsplit=1)
    mod = import_module(mod_path)
    fn = getattr(mod, fn_name)
    return fn

def main():
    parser = obs.default_argparse(6400)
    parser.add_argument('--get_config', type=str, default='config.get_config',
        help="extract ansatz from config; get_config(nelec: int, rs: float)")
    parser.add_argument('--fchk', type=str, default='checkpoint.pkl')
    parser.add_argument('--nx', '-n', type=int, default=48)
    args = parser.parse_args()
    fchk = args.fchk

    # set up folder to cache results
    tname = os.path.basename(args.ftraj)
    cache_dir = 'cache-' + tname[:tname.rfind('.')] + '-obs'

    # read metadata
    meta_sys = obs.read_meta(args.fyml)
    cell = jnp.asarray(meta_sys['system']['cell'])
    nelec = meta_sys['system']['nelec']
    print('nelec = ', nelec)
    spins = meta_sys['system']['spins']
    print('spins = ', spins)
    meta_sys = dict(spins=list(spins), cell=cell.tolist())

    # derived metadata
    ndim = len(cell)
    rs = heg_rs(cell, nelec)

    # read trajectories
    traj, straj = obs.read_traj(args.ftraj, ndim, nelec)
    niter, nwalker, nelec, ndim = traj.shape
    print('initial: ', traj.shape)
    traj = obs.reshape_traj(traj[args.iter:args.jter],
        args.batch_size, max_batch=args.max_batch)
    straj = obs.reshape_traj(straj[args.iter:args.jter],
        args.batch_size, max_batch=args.max_batch)
    print('reshape: ', traj.shape)

    get_config = import_function(args.get_config)
    cfg = get_config(nelec, rs)#, twist)
    ansatz = cfg.ansatz
    # read ansatz parameters
    params = load_pickle(fchk)[1]

    # calculate observables
    bins = (args.nx,)*ndim
    calc_mdens = obs.gen_calc_mdens(cell, bins, ansatz, params)
    meta_mdens, ym, ye = obs.calc_obs((traj, straj), calc_mdens)
    edges = meta_mdens['edges']
    meta = dict(aname='mdens')
    meta.update(meta_sys)
    for i, e in enumerate(edges):
        meta['edge%d' % i] = e.tolist()
    obs.save_obs('%s/mdens' % cache_dir, meta, ym, ye)

if __name__ == '__main__':
     main()  # set no global variable
