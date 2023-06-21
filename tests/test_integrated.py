import pytest
import os
import jax
from jax import numpy as jnp

import fvmc


@pytest.mark.veryslow
def test_h2_kfac(tmp_path, capfd):

    cfg = fvmc.config.default()
    cfg.seed = 0
    cfg.verbosity = "INFO"
    #cfg.restart.params = 'oldstates.pkl'

    cfg.system.nuclei = [[0.,0.,0.], [0.,0.,1.]]
    cfg.system.elems = [1., 1.]

    #TODO make parameters for ansatz

    cfg.sample.size = 2048
    cfg.sample.sampler = 'mala'
    cfg.sample.mala.tau = 0.1
    cfg.sample.mala.steps = 10
    cfg.sample.burn_in = 100

    cfg.optimize.iterations = 100
    cfg.optimize.loss.energy_clipping = 5.
    cfg.optimize.optimizer = 'kfac'
    cfg.optimize.lr.base = 1e-4

    cfg.log.stat_every = 50

    os.chdir(tmp_path)
    train_state = fvmc.train.main(cfg)

    for fname in (cfg.log.stat_path, cfg.log.ckpt_path, cfg.log.hpar_path):
        assert (tmp_path / fname).exists()

    assert jax.tree_util.tree_all(jax.tree_map(lambda a: jnp.all(~jnp.isnan(a)), train_state))