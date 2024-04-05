import pytest
import os
import jax
from jax import numpy as jnp

import fvmc


def get_shared_cfg():
    cfg = fvmc.config.default()
    cfg.seed = 0
    cfg.verbosity = "INFO"
    #cfg.restart.params = 'oldstates.pkl'

    #TODO make parameters for ansatz

    cfg.sample.size = 512
    cfg.sample.burn_in = 100

    cfg.loss.energy_clipping = 5.

    cfg.optimize.iterations = 100
    cfg.optimize.optimizer = 'kfac'
    cfg.optimize.lr.base = 1e-4

    cfg.log.stat_every = 50
    cfg.log.dump_every = 50

    return cfg


def shared_test(tmp_path, cfg):
    os.chdir(tmp_path)
    train_state = fvmc.train.main(cfg)

    for fname in (cfg.log.stat_path, cfg.log.ckpt_path, cfg.log.hpar_path, cfg.log.dump_path):
        assert (tmp_path / fname).exists()
    if cfg.log.use_tensorboard:
        assert (tmp_path / cfg.log.tracker_path).exists()

    assert jax.tree_util.tree_all(jax.tree_map(lambda a: jnp.all(~jnp.isnan(a)), train_state))


@pytest.mark.veryslow
def test_h2_kfac(tmp_path):

    cfg = get_shared_cfg()
    cfg.system.nuclei = [[0.,0.,0.], [0.,0.,1.]]
    cfg.system.elems = [1., 1.]

    cfg.sample.sampler = 'mala'
    cfg.sample.mala.tau = 0.1
    cfg.sample.mala.steps = 10

    shared_test(tmp_path, cfg)


@pytest.mark.veryslow
def test_heg_adam(tmp_path):

    cfg = get_shared_cfg()
    cfg.system.nuclei = None
    cfg.system.elems = None
    cfg.system.cell = [1., 1., 1.]
    cfg.system.charge = -2

    cfg.sample.sampler = 'mcmc'
    cfg.sample.mcmc.sigma = 0.1
    cfg.sample.mcmc.steps = 10

    cfg.optimize.optimizer = 'adabelief'
    cfg.optimize.grad_clipping = 0.01

    shared_test(tmp_path, cfg)


def get_heg_cfg():
    import numpy as np

    rs = 30
    n_elec = 12
    sig = 2.6 * rs**(3/4)
    cell = np.diag([1.714163052355123114e+02, 1.979344999424281752e+02])
    wgrid = np.mgrid[:3, :4].transpose(1, 2, 0).reshape(-1, 2)
    wcpos = wgrid * np.array([1/3, 1/4]) @ cell

    def conf_init_fn(key):
        return wcpos + 0.1 * sig * jax.random.normal(key, wcpos.shape)

    cfg = fvmc.config.default()
    cfg.seed = 42
    cfg.verbosity = "INFO"
    # <system>
    cfg.system.charge = -n_elec
    cfg.system.elems = None
    cfg.system.nuclei = None
    cfg.system.cell = cell
    # <sample>
    cfg.sample.size = 128
    cfg.sample.burn_in = 10
    cfg.sample.sampler = 'mala'
    cfg.sample.mala.tau = 0.02 * rs
    cfg.sample.mala.steps = 20
    cfg.sample.mala.grad_clipping = 1.
    cfg.sample.conf_init_fn = conf_init_fn
    # <energy>
    cfg.loss.energy_clipping = 5.
    cfg.loss.clip_from_median = True
    cfg.loss.ke_kwargs.forward_mode = True
    cfg.loss.ke_kwargs.partition_size = 1
    # <optimize>
    cfg.optimize.iterations = 10
    cfg.optimize.optimizer = 'sr'
    cfg.optimize.lr.base = 1.
    cfg.optimize.lr.decay_time = 10
    cfg.optimize.lr.warmup_steps = 5
    cfg.optimize.sr.max_norm = 1.
    cfg.optimize.sr.proximal = 0.9
    # <output>
    cfg.log.stat_every = 1
    cfg.log.dump_every = 5
    cfg.log.ckpt_every = 5
    cfg.log.use_tensorboard = False

    return cfg


@pytest.mark.veryslow
def test_heg_nnbf(tmp_path):
    from fvmc.wavefunction.heg import PlanewaveSlater, PairJastrowCCK, ElecProductModel
    from fvmc.wavefunction.hegnn import NeuralBackflow

    cfg = get_heg_cfg()
    nelec = -cfg.system.charge
    spins = [nelec // 2, nelec - nelec // 2]
    cell = cfg.system.cell
    npw = 3 * nelec
    ansatz = ElecProductModel(
        [
            PlanewaveSlater(spins=spins, cell=cell, n_k=npw),
            PairJastrowCCK(spins=spins, cell=cell)
        ],
        backflow=NeuralBackflow(spins=spins,cell=cell),
        # apply_backflow=[True, False],
    )
    cfg.ansatz = ansatz

    shared_test(tmp_path, cfg)
