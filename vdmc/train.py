import jax
from jax import numpy as jnp
from ml_collections import ConfigDict
from tensorboardX import SummaryWriter
from typing import NamedTuple, Tuple

from . import LOGGER
from .utils import PyTree, Array
from .utils import Printer, save_checkpoint, load_pickle, cfg_to_yaml
from .utils import paxis
from .wavefunction import make_jastrow_slater
from .sampler import make_sampler, make_batched, make_multistep
from .estimator import make_eval_local, make_eval_total
from .optimizer import make_optimizer, make_lr_schedule


class SysInfo(NamedTuple):
    ions: Array
    elems: Array
    n_elec: Tuple[int, int]


class TrainingState(NamedTuple):
    key: Array
    params: PyTree
    mc_state: PyTree
    opt_state: PyTree


def prepare(system_cfg, ansatz_cfg, sample_cfg, optimize_cfg, key=None, restart_cfg=None):
    """prepare system, ansatz, sampler, optimizer and training state"""

    # make sure all cfg are ConfigDict
    system_cfg, ansatz_cfg, sample_cfg, optimize_cfg, restart_cfg= \
        map(ConfigDict, (system_cfg, ansatz_cfg, sample_cfg, optimize_cfg, restart_cfg))

    # parse system, may be changed (e.g. using pyscf mol)
    ions = jnp.asarray(system_cfg.ions)
    elems = jnp.asarray(system_cfg.elems)
    tot_elec = int(sum(elems) + system_cfg.charge)
    spin = system_cfg.spin if system_cfg.spin is not None else tot_elec % 2
    assert (tot_elec - spin) % 2 == 0, \
        f"system with {tot_elec} electrons cannot have spin {spin}"
    n_elec = (tot_elec + spin) // 2, (tot_elec - spin) // 2
    system = SysInfo(ions, elems, n_elec)

    # make wavefunction
    ansatz = make_jastrow_slater(ions, elems, spin, **ansatz_cfg)
    
    # make estimators
    local_fn = make_eval_local(ansatz, ions, elems)
    loss_fn = make_eval_total(local_fn, optimize_cfg.energy_clipping)
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    # make sampler
    n_sample = sample_cfg.size
    n_chain = sample_cfg.chains or n_sample
    if n_chain % n_sample != 0:
        LOGGER.warning("Sample size not divisible by batch size, rounding up")
    n_multistep = -(-n_sample // n_chain)
    n_batch = n_chain # TODO when parallel, this is the chains on a local device
    raw_sampler = make_sampler(ansatz, tot_elec, **sample_cfg.sampler)
    sampler = make_multistep(raw_sampler, n_step=n_multistep, concat=False)
    sampler = make_batched(sampler, n_batch=n_batch, concat=True)
    sampler = jax.tree_map(jax.jit, sampler) # TODO make it pmapped when parallel

    # make optimizer
    lr_schedule = make_lr_schedule(**optimize_cfg.lr)
    optimizer = make_optimizer(
        loss_and_grad, 
        lr_schedule=lr_schedule,
        value_func_has_aux=True,
        value_func_has_rng=False,
        value_func_has_state=False,
        multi_device=False, # TODO this is true when parallel
        **optimize_cfg.optimizer)

    # make training states 
    # TODO add logging because these can be slow
    # TODO parallel initialize is different
    if "states" in restart_cfg and restart_cfg.states:
        LOGGER.info("Loading parameters and states from saved file")
        train_state = TrainingState(*load_pickle(restart_cfg.states))
    else:
        LOGGER.info("Initializing parameters and states")
        assert key is not None, \
            "key is required if not restarting from previous state"
        key, parkey, mckey, optkey, statekey = jax.random.split(key, 5)
        # initialize params
        if "params" in restart_cfg and restart_cfg.params:
            LOGGER.info("Loading parameters from saved file")
            params = load_pickle(restart_cfg.params)
            if isinstance(params, tuple): params = params[1]
        else:
            fake_input = jnp.zeros((tot_elec, 3))
            params = ansatz.init(parkey, fake_input)
        # initialize mc state
        mc_state = sampler.init(mckey, params)
        if "burn_in" in sample_cfg and sample_cfg.burn_in > 0:
            LOGGER.info(f"Burning in the sampler for {sample_cfg.burn_in} steps")
            key, subkey = jax.random.split(key)
            mc_state = sampler.burn_in(subkey, params, mc_state, sample_cfg.burn_in)
        key, subkey = jax.random.split(key)
        mc_state, init_data, _ = sampler.sample(key, params, mc_state)
        # initialize opt state
        opt_state = optimizer.init(params, optkey, init_data)
        # assemble training state
        train_state = TrainingState(statekey, params, mc_state, opt_state)

    return system, ansatz, loss_fn, sampler, optimizer, train_state


def gen_training_step(sampler, optimizer):
    """generate a training loop step function from sampler and optimizer"""

    def training_step(train_state):
        key, params, mc_state, opt_state = train_state
        key, mckey, optkey = jax.random.split(key, 3)
        mc_state, data, mc_info = sampler.sample(mckey, params, mc_state)
        params, opt_state, opt_info = optimizer.step(params, opt_state, optkey, batch=data)
        mc_state = sampler.refresh(mc_state, params)
        return TrainingState(key, params, mc_state, opt_state), (mc_info, opt_info)

    return training_step


def train(step_fn, train_state, iterations, log_cfg):
    """run the optimization loop (sample + update)"""

    log_cfg = ConfigDict(log_cfg)
    writer = SummaryWriter(log_cfg.stat_path)
    print_fields = {"step": "", "e_tot": ".4f", 
                    "avg_s": ".4f", "var_e": ".3e", 
                    "acc": ".2f", "lr": ".2e"} 
    printer = Printer(print_fields, time_format=".2f")

    LOGGER.info("Start training")
    printer.print_header("# ")
    for ii in range(iterations + 1):
        printer.reset_timer()

        # main training loop
        train_state, (mc_info, opt_info) = step_fn(train_state)

        # log stats
        if ii % log_cfg.stat_every == 0:
            acc_rate = paxis.all_mean(mc_info["is_accepted"])
            stat_dict = {"step": opt_info["step"]-1, **opt_info["aux"], 
                         "acc": acc_rate, "lr":opt_info["learning_rate"]}
            printer.print_fields(stat_dict)
            for k,v in stat_dict.items():
                writer.add_scalar(k, v, ii)
        
        # checkpoint
        if ii % log_cfg.ckpt_every == 0:
            save_checkpoint(log_cfg.ckpt_path, tuple(train_state), 
                            keep=log_cfg.ckpt_keep)
    
    writer.close()
    return train_state


def main(cfg):
    cfg = ConfigDict(cfg)

    if "hpar_path" in cfg.log:
            with open(cfg.log.hpar_path, "w") as hpfile:
                print(cfg_to_yaml(cfg), file=hpfile)

    import logging
    logging_level = getattr(logging, cfg.logging_level.upper())
    LOGGER.setLevel(logging_level)

    key = jax.random.PRNGKey(cfg.seed) if 'seed' in cfg else None
    system, ansatz, loss_fn, sampler, optimizer, train_state \
        = prepare(cfg.system, cfg.ansatz, cfg.sample, cfg.optimize, key, cfg.restart)

    training_step = gen_training_step(sampler, optimizer)
    train_state = train(training_step, train_state, cfg.optimize.iterations, cfg.log)
    
    return train_state