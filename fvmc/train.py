from functools import partial
from typing import NamedTuple, Optional, Tuple, Union

import jax
import kfac_jax
import numpy as onp
from flax import linen as nn
from jax import numpy as jnp
from ml_collections import ConfigDict
from tensorboardX import SummaryWriter

from . import LOGGER
from .estimator import (build_eval_local_elec, build_eval_local_full,
                        build_eval_total)
from .neuralnet import FermiNet
from .optimizer import build_lr_schedule, build_optimizer
from .sampler import (build_conf_init_fn, build_sampler, make_batched,
                      make_multistep)
from .utils import (PAXIS, Array, ArrayTree, Printer, PyTree, adaptive_split,
                    cfg_to_yaml, load_pickle, multi_process_name,
                    save_checkpoint)
from .wavefunction import (FixNuclei, NucleiGaussianSlater, ProductModel,
                           build_jastrow_slater, log_prob_from_model)


class SysInfo(NamedTuple):
    elems: Array
    nuclei: Array
    n_elec: Tuple[int, int]


class TrainingState(NamedTuple):
    key: Array
    params: ArrayTree
    mc_state: ArrayTree
    opt_state: ArrayTree


def trim_training_state(train_state):
    key, params, mc_state, opt_state = train_state
    if key.ndim > 1: # pmapped
        params, opt_state = jax.tree_map(lambda x: x[0], (params, opt_state))
    return TrainingState(key, params, mc_state, opt_state)


def match_loaded_state_to_device(train_state, multi_device: bool):
    n_local_device = jax.local_device_count()
    key, params, mc_state, opt_state = train_state
    n_paxis = key.ndim - 1
    if not multi_device: # to single device
        if n_paxis == 0: # from single device
            return train_state
        else: # from multi device
            key = key[0]
            mc_state = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), mc_state)
    else: # to multi device
        params, opt_state = \
            kfac_jax.utils.replicate_all_local_devices((params, opt_state))
        if n_paxis > 0 and key.shape[0] >= n_local_device:
            key = key[:n_local_device]
        else:
            key = jax.random.split(key.reshape(-1, 2)[0], n_local_device)
        mc_state = jax.tree_map(
            lambda x: x.reshape(n_local_device, -1, *x.shape[n_paxis+1:]),
            mc_state)
        key, mc_state = kfac_jax.utils.broadcast_all_local_devices((key, mc_state))
    return TrainingState(key, params, mc_state, opt_state)


def prepare(system_cfg, ansatz_cfg, sample_cfg, optimize_cfg,
            fully_quantum=False, key=None, restart_cfg=None, multi_device=False):
    """prepare system, ansatz, sampler, optimizer and training state"""

    # handle multi device settings
    n_device = jax.device_count() if multi_device else 1
    rng_split = partial(adaptive_split, multi_device=multi_device)
    LOGGER.info("local device: %d, global devices: %d, process id: %d",
        jax.local_device_count(), n_device, jax.process_index())

    # make sure all cfg are ConfigDict
    system_cfg, sample_cfg, optimize_cfg, restart_cfg= \
        map(ConfigDict, (system_cfg, sample_cfg, optimize_cfg, restart_cfg))

    # parse system, may be changed (e.g. using pyscf mol)
    nuclei = jnp.asarray(system_cfg.nuclei)
    elems = jnp.asarray(system_cfg.elems)
    tot_elec = int(sum(elems) + system_cfg.charge)
    spin = system_cfg.spin if system_cfg.spin is not None else tot_elec % 2
    assert (tot_elec - spin) % 2 == 0, \
        f"system with {tot_elec} electrons cannot have spin {spin}"
    n_elec = (tot_elec + spin) // 2, (tot_elec - spin) // 2
    system = SysInfo(elems, nuclei, n_elec)

    # make wavefunction
    if isinstance(ansatz_cfg, nn.Module):
        ansatz = ansatz_cfg
    else:
        ansatz_cfg = ConfigDict(ansatz_cfg)
        # ansatz = build_jastrow_slater(elems, nuclei, spin, 
        #     dynamic_nuclei=fully_quantum, **ansatz_cfg)
        ansatz = FermiNet(elems=elems, spin=spin, **ansatz_cfg)
        if fully_quantum:
            ansatz = ProductModel([ansatz, NucleiGaussianSlater(nuclei, 0.1)])
        else:
            ansatz = FixNuclei(ansatz, nuclei)
    log_prob_fn = log_prob_from_model(ansatz)

    # make estimators
    local_fn = (build_eval_local_full(ansatz, elems) if fully_quantum
                else build_eval_local_elec(ansatz, elems, nuclei))
    loss_fn = build_eval_total(local_fn, 
        pmap_axis_name=PAXIS.name, **optimize_cfg.loss)
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    # make sampler
    n_sample = sample_cfg.size
    n_chain = sample_cfg.chains or n_sample
    if n_chain % n_sample != 0:
        LOGGER.warning("Sample size not divisible by batch size, rounding up")
    n_multistep = -(-n_sample // n_chain)
    n_batch = n_chain // n_device
    conf_init_fn = build_conf_init_fn(
        elems, nuclei, tot_elec, with_r=fully_quantum)
    raw_sampler = build_sampler(
        log_prob_fn, 
        conf_init_fn, 
        name=sample_cfg.sampler,
        **sample_cfg.get(sample_cfg.sampler, {}))
    sampler = make_multistep(raw_sampler, n_step=n_multistep, concat=False)
    sampler = make_batched(sampler, n_batch=n_batch, concat=True)
    sampler = jax.tree_map(PAXIS.pmap if multi_device else jax.jit, sampler)

    # make optimizer
    lr_schedule = build_lr_schedule(**optimize_cfg.lr)
    optimizer = build_optimizer(
        loss_and_grad, 
        name=optimize_cfg.optimizer,
        lr_schedule=lr_schedule,
        value_func_has_aux=True,
        value_func_has_rng=False,
        value_func_has_state=False,
        multi_device=multi_device,
        pmap_axis_name=PAXIS.name,
        log_prob_func=log_prob_fn,
        grad_clipping=optimize_cfg.get("grad_clipping", None),
        **optimize_cfg.get(optimize_cfg.optimizer, {}))

    # make training states 
    if "states" in restart_cfg and restart_cfg.states:
        LOGGER.info("Loading parameters and states from saved file")
        state_path = multi_process_name(restart_cfg.states)
        train_state = TrainingState(*load_pickle(state_path))
        train_state = match_loaded_state_to_device(train_state, multi_device)
    else:
        LOGGER.info("Initializing parameters and states")
        assert key is not None, \
            "key is required if not restarting from previous state"
        # initialize params
        if "params" in restart_cfg and restart_cfg.params:
            LOGGER.info("Loading parameters from saved file")
            param_path = multi_process_name(restart_cfg.params)
            params = load_pickle(param_path)
            if isinstance(params, tuple): params = params[1]
        else:
            key, parkey, skey = jax.random.split(key, 3) # init use single key
            fake_input = conf_init_fn(skey)
            params = (ansatz.init(parkey, *fake_input) if fully_quantum 
                      else ansatz.init(parkey, fake_input))
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
    if jax.process_index() == 0:
        writer = SummaryWriter(log_cfg.stat_path)
    print_fields = {"step": "", 
                    "e_tot": ".4f", "std_e": ".3e", 
                    "acc": ".2f", "lr": ".2e"} 
    printer = Printer(print_fields, time_format=".2f")

    # mysterious step to prevent kfac memory error
    train_state = jax.tree_map(jnp.copy, train_state)

    LOGGER.info("Start training")
    if jax.process_index() == 0:
        printer.print_header("# ")

    for ii in range(iterations):
        printer.reset_timer()

        # main training loop
        train_state, (mc_info, opt_info) = step_fn(train_state)

        if not jax.tree_util.tree_all(
          jax.tree_map(lambda a: jnp.all(~jnp.isnan(a)), train_state)):
            raise ValueError(f"NaN found in training state at step {ii} "
                             f"(log step {opt_info['step']-1})")
        if opt_info["aux"]["nans"] > 0:
            LOGGER.warning("%d NaN found in local energy at step %d (log step %d)",
                           opt_info["aux"]["nans"], ii, opt_info['step']-1)

        # log stats
        if ((ii % log_cfg.stat_every == 0 or ii == iterations-1) 
          and jax.process_index() == 0): # only print for process 0 (all same)
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
            ckpt_path = multi_process_name(log_cfg.ckpt_path)
            save_checkpoint(
                ckpt_path, 
                tuple(trim_training_state(train_state)),
                keep=log_cfg.ckpt_keep)

    if jax.process_index() == 0:
        writer.close()
    return train_state


def main(cfg):
    cfg = ConfigDict(cfg)

    if "hpar_path" in cfg.log and jax.process_index() == 0:
        with open(cfg.log.hpar_path, "w") as hpfile:
            print(cfg_to_yaml(cfg), file=hpfile)

    import logging
    verbosity = getattr(logging, cfg.verbosity.upper())
    LOGGER.setLevel(verbosity)

    key = jax.random.PRNGKey(cfg.seed) if 'seed' in cfg else None
    system, ansatz, loss_fn, sampler, optimizer, train_state \
        = prepare(cfg.system, cfg.ansatz, cfg.sample, cfg.optimize, 
                  cfg.fully_quantum, key, cfg.restart, cfg.multi_device)

    training_step = build_training_step(sampler, optimizer)
    train_state = run(training_step, train_state, cfg.optimize.iterations, cfg.log)
    
    return train_state