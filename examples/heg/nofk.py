#!/usr/bin/env python3
import os
import numpy as np
import jax
import jax.numpy as jnp
from importlib import import_module
from typing import Callable
from fvmc import observable as obs
from fvmc.wavefunction.heg import heg_rs
from fvmc.utils import load_pickle
from fvmc.train import TrainingState

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
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('ftraj', type=str)
    parser.add_argument('--fyml', type=str, default='hparams.yaml')
    parser.add_argument('--get_config', type=str, default='config.get_config',
        help="extract ansatz from config; get_config(nelec: int, rs: float)")
    parser.add_argument('--iter', '-i', type=int, default=0)
    parser.add_argument('--jter', '-j', type=int, default=None)
    parser.add_argument('--fchk', type=str, default='checkpoint.pkl')
    parser.add_argument('--max_batch', '-m', type=int, default=64,
        help='reduce to save time')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
        help='reduce to save memory')
    parser.add_argument('--tx', type=float, default=0.0)
    parser.add_argument('--ty', type=float, default=0.0)
    parser.add_argument('--kcut', type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    nbatch = args.batch_size  # reduce to save memory
    fchk = args.fchk
    kcut = args.kcut
    twist = jnp.array([args.tx, args.ty])
    key = jax.random.PRNGKey(args.seed)

    # set up folder to cache results
    tname = os.path.basename(args.ftraj)
    cache_dir = 'cache-' + tname[:tname.rfind('.')] + '-obs'

    # read metadata
    meta_sys = obs.read_meta(args.fyml)
    cell = jnp.asarray(meta_sys['system']['cell'])
    nelec = meta_sys['system']['nelec']
    spins = meta_sys['system']['spins']
    print('spins = ', spins)
    meta_sys = dict(spins=list(spins), cell=cell.tolist())

    # derived metadata
    ndim = len(cell)
    rs = heg_rs(cell, nelec)
    if kcut is None:
        nspin = len(spins)
        kf = 2./rs/nspin**0.5
        kcut = 4*kf

    # read trajectories
    traj = obs.read_traj(args.ftraj, ndim)[args.iter:args.jter]
    niter, nwalker, nelec, ndim = traj.shape
    print('initial: ', traj.shape)
    traj = traj.reshape(-1, nbatch, nelec, ndim)
    print('reshape: ', traj.shape)
    nbatch_tot = len(traj)
    nevery = nbatch_tot//args.max_batch
    traj = traj[::nevery]
    print('max batch: ', traj.shape)

    get_config = import_function(args.get_config)
    cfg = get_config(nelec, rs)#, twist)
    ansatz = cfg.ansatz
    # read ansatz parameters
    train_state = TrainingState(*load_pickle(fchk))
    params = train_state.params

    # calculate observables
    calc_nofk = obs.gen_calc_nofk(cell, kcut, twist, nelec, ansatz, params, key)
    meta_nofk, nkm, nke = obs.calc_obs(traj, calc_nofk)
    meta = dict(aname='nofk', kvecs=meta_nofk['kvecs'].tolist())
    meta.update(meta_sys)
    obs.save_obs('%s/nofk' % cache_dir, meta, nkm, nke)

if __name__ == '__main__':
  main()  # set no global variable
