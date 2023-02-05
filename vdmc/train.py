import jax
import kfac_jax
from jax import numpy as jnp
from ml_collections import ConfigDict
from tensorboardX import SummaryWriter
from typing import NamedTuple, Tuple
from functools import partial

from . import LOGGER
from .utils import PyTree, Array
from .utils import Printer, save_checkpoint, load_pickle, cfg_to_yaml
from .utils import PAXIS, adaptive_split
from .wavefunction import build_jastrow_slater
from .sampler import build_sampler, make_batched, make_multistep
from .estimator import build_eval_local, build_eval_total
from .optimizer import build_optimizer, build_lr_schedule


class SysInfo(NamedTuple):
    ions: Array
    elems: Array
    n_elec: Tuple[int, int]


class TrainingState(NamedTuple):
    key: Array
    params: PyTree
    mc_state: PyTree
    opt_state: PyTree


def prepare(system_cfg, ansatz_cfg, sample_cfg, optimize_cfg, 
            key=None, restart_cfg=None, multi_device=False):
    """prepare system, ansatz, sampler, optimizer and training state"""

    # handle multi device settings
    n_device = jax.device_count() if multi_device else 1
    rng_split = partial(adaptive_split, multi_device=multi_device)

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
    ansatz = build_jastrow_slater(ions, elems, spin, **ansatz_cfg)
    
    # make estimators
    local_fn = build_eval_local(ansatz, ions, elems)
    loss_fn = build_eval_total(local_fn, optimize_cfg.energy_clipping, PAXIS.name)
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    # make sampler
    n_sample = sample_cfg.size
    n_chain = sample_cfg.chains or n_sample
    if n_chain % n_sample != 0:
        LOGGER.warning("Sample size not divisible by batch size, rounding up")
    n_multistep = -(-n_sample // n_chain)
    n_batch = n_chain // n_device
    raw_sampler = build_sampler(ansatz, tot_elec, **sample_cfg.sampler)
    sampler = make_multistep(raw_sampler, n_step=n_multistep, concat=False)
    sampler = make_batched(sampler, n_batch=n_batch, concat=True)
    sampler = jax.tree_map(PAXIS.pmap if multi_device else jax.jit, sampler)

    # make optimizer
    lr_schedule = build_lr_schedule(**optimize_cfg.lr)
    optimizer = build_optimizer(
        loss_and_grad, 
        lr_schedule=lr_schedule,
        value_func_has_aux=True,
        value_func_has_rng=False,
        value_func_has_state=False,
        multi_device=multi_device,
        pmap_axis_name=PAXIS.name,
        **optimize_cfg.optimizer)

    # make training states 
    if "states" in restart_cfg and restart_cfg.states:
        LOGGER.info("Loading parameters and states from saved file")
        train_state = TrainingState(*load_pickle(restart_cfg.states))
    else:
        LOGGER.info("Initializing parameters and states")
        assert key is not None, \
            "key is required if not restarting from previous state"
        # initialize params
        if "params" in restart_cfg and restart_cfg.params:
            LOGGER.info("Loading parameters from saved file")
            params = load_pickle(restart_cfg.params)
            if isinstance(params, tuple): params = params[1]
        else:
            key, parkey = jax.random.split(key) # init use single key
            fake_input = jnp.zeros((tot_elec, 3))
            params = ansatz.init(parkey, fake_input)
            if multi_device:
                params = kfac_jax.utils.replicate_all_local_devices(params)
        # after init params, use pmapped key
        if multi_device:
            key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
        key, mckey, optkey, statekey = rng_split(key, 4)
        # initialize mc state
        mc_state = sampler.init(mckey, params)
        if "burn_in" in sample_cfg and sample_cfg.burn_in > 0:
            LOGGER.info(f"Burning in the sampler for {sample_cfg.burn_in} steps")
            key, subkey = rng_split(key)
            mc_state = sampler.burn_in(subkey, params, mc_state, sample_cfg.burn_in)
        key, subkey = rng_split(key)
        mc_state, init_data, _ = sampler.sample(subkey, params, mc_state)
        # initialize opt state
        opt_state = optimizer.init(params, optkey, init_data)
        # assemble training state
        train_state = TrainingState(statekey, params, mc_state, opt_state)

    return system, ansatz, loss_fn, sampler, optimizer, train_state


def build_training_step(sampler, optimizer):
    """generate a training loop step function from sampler and optimizer"""

    rng_split = partial(adaptive_split, multi_device=optimizer.multi_device)

    def training_step(train_state):
        key, params, mc_state, opt_state = train_state
        key, mckey, optkey = rng_split(key, 3)
        mc_state, data, mc_info = sampler.sample(mckey, params, mc_state)
        params, opt_state, opt_info = optimizer.step(params, opt_state, optkey, batch=data)
        mc_state = sampler.refresh(mc_state, params)
        return TrainingState(key, params, mc_state, opt_state), (mc_info, opt_info)

    return training_step


def run(step_fn, train_state, iterations, log_cfg):
    """run the optimization loop (sample + update)"""

    log_cfg = ConfigDict(log_cfg)
    writer = SummaryWriter(log_cfg.stat_path)
    print_fields = {"step": "", "e_tot": ".4f", 
                    "avg_s": ".4f", "var_e": ".3e", 
                    "acc": ".2f", "lr": ".2e"} 
    printer = Printer(print_fields, time_format=".2f")

    # mysterious step to prevent kfac memory error
    train_state = jax.tree_map(jnp.copy, train_state)

    LOGGER.info("Start training")
    printer.print_header("# ")
    for ii in range(iterations):
        printer.reset_timer()

        # main training loop
        train_state, (mc_info, opt_info) = step_fn(train_state)

        # log stats
        if ii % log_cfg.stat_every == 0 or ii == iterations-1:
            acc_rate = PAXIS.all_mean(mc_info["is_accepted"])
            stat_dict = {"step": opt_info["step"]-1, **opt_info["aux"], 
                         "acc": acc_rate, "lr":opt_info["learning_rate"]}
            stat_dict = jax.tree_map( # collect from potential pmap
                lambda x: x[0] if jnp.ndim(x) > 0 else x, stat_dict)
            printer.print_fields(stat_dict)
            for k,v in stat_dict.items():
                writer.add_scalar(k, v, ii)
        
        # checkpoint
        if ii % log_cfg.ckpt_every == 0 or ii == iterations-1:
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
    verbosity = getattr(logging, cfg.verbosity.upper())
    LOGGER.setLevel(verbosity)

    key = jax.random.PRNGKey(cfg.seed) if 'seed' in cfg else None
    system, ansatz, loss_fn, sampler, optimizer, train_state \
        = prepare(cfg.system, cfg.ansatz, cfg.sample, cfg.optimize, 
                  key, cfg.restart, cfg.multi_device)

    training_step = build_training_step(sampler, optimizer)
    train_state = run(training_step, train_state, cfg.optimize.iterations, cfg.log)
    
    return train_state