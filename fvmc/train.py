import os
from functools import partial
from typing import NamedTuple, Optional, Sequence, Tuple

import jax
import kfac_jax
import numpy as onp
from flax import linen as nn
from jax import numpy as jnp
from ml_collections import ConfigDict

from . import LOGGER
from .estimator import (build_eval_local_elec, build_eval_local_full,
                        build_eval_total)
from .optimizer import build_lr_schedule, build_optimizer
from .sampler import (build_conf_init_fn, build_sampler, make_batched,
                      make_multistep)
from .utils import (PAXIS, Array, ArrayTree, Printer, PyTree, adaptive_split,
                    backup_if_exist, cfg_to_yaml, load_pickle,
                    multi_process_name, save_checkpoint)
from .wavefunction import (FermiNet, FermiNetPbc, FixNuclei,
                           NucleiGaussianSlater, NucleiGaussianSlaterPbc,
                           ProductModel, build_jastrow_slater,
                           log_prob_from_model, log_psi_from_model)


class SysInfo(NamedTuple):
    elems: Sequence[int]
    spins: Tuple[int, int]
    nuclei: Array
    cell: Optional[Array] = None


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


def parse_system_cfg(system_cfg):
    # optional cell
    cell = system_cfg.get("cell", None)
    if cell is not None:
        cell = jnp.asarray(cell)
        if cell.ndim == 1:
            cell = jnp.diag(cell)
    # no nuclei case, HEG, has to be in pbc
    if system_cfg.nuclei is None and system_cfg.elems is None:
        assert cell is not None
        space_dim = cell.shape[-1]
        nuclei = jnp.zeros((0, space_dim), dtype=float)
        elems = jnp.zeros((0,), dtype=int)
    else:
        nuclei = jnp.asarray(system_cfg.nuclei)
        elems = jnp.asarray(system_cfg.elems, dtype=int)
    tot_elec = int(sum(elems) - system_cfg.charge)
    spin = system_cfg.spin if system_cfg.spin is not None else tot_elec % 2
    assert (tot_elec - spin) % 2 == 0, \
        f"system with {tot_elec} electrons cannot have spin {spin}"
    n_elec = (tot_elec + spin) // 2, (tot_elec - spin) // 2
    system = SysInfo(elems, n_elec, nuclei, cell)
    return system


def prepare(system_cfg, ansatz_cfg, sample_cfg, loss_cfg, optimize_cfg,
            quantum_nuclei=False, key=None, restart_cfg=None, multi_device=False):
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
    system = parse_system_cfg(system_cfg)
    elems, n_elec, nuclei, cell = system

    # make wavefunction
    if all(callable(getattr(ansatz_cfg, _f, None)) for _f in ('init', 'apply')):
        ansatz = ansatz_cfg
    else:
        ansatz_cfg = ConfigDict(ansatz_cfg)
        # ansatz = build_jastrow_slater(elems, n_elec, nuclei,
        #     dynamic_nuclei=quantum_nuclei, **ansatz_cfg)
        if quantum_nuclei:
            if cell is None:
                elec_ansatz = FermiNet(elems=elems, spins=n_elec, **ansatz_cfg)
                nuclei_ansatz = NucleiGaussianSlater(nuclei, 0.1)
                ansatz = ProductModel([elec_ansatz, nuclei_ansatz])
            else:
                nuclei_part = NucleiGaussianSlaterPbc(cell, nuclei, 0.1)
                ansatz = FermiNetPbc(elems=elems, spins=n_elec, cell=cell,
                                     nuclei_module=nuclei_part, **ansatz_cfg)
        else:
            if cell is None:
                raw_ansatz = FermiNet(elems=elems, spins=n_elec, **ansatz_cfg)
            else:
                raw_ansatz = FermiNetPbc(elems=elems, spins=n_elec,
                                         cell=cell, **ansatz_cfg)
            ansatz = FixNuclei(raw_ansatz, nuclei)
    log_prob_fn = log_prob_from_model(ansatz)
    log_psi_fn = log_psi_from_model(ansatz)

    # make estimators
    loss_cfg = dict(loss_cfg) # so that we can pop
    ke_kwargs = loss_cfg.pop("ke_kwargs", {})
    pe_kwargs = loss_cfg.pop("pe_kwargs", {})
    extpots = system_cfg.get("external_potentials", {})
    lclargs = dict(ke_kwargs=ke_kwargs, pe_kwargs=pe_kwargs,
                   extpots=extpots, stop_gradient=True)
    local_fn = (build_eval_local_full(ansatz, elems, cell, **lclargs)
                if quantum_nuclei else
                build_eval_local_elec(ansatz, elems, nuclei, cell, **lclargs))
    loss_fn = build_eval_total(local_fn,
        pmap_axis_name=PAXIS.name, **loss_cfg)
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    # make sampler
    n_sample = sample_cfg.size
    n_chain = sample_cfg.chains or n_sample
    if n_sample % n_chain != 0:
        LOGGER.warning("Sample size not divisible by batch size, rounding up")
    n_multistep = -(-n_sample // n_chain)
    n_batch = n_chain // n_device
    conf_init_fn = sample_cfg.get("conf_init_fn", build_conf_init_fn(
        elems, nuclei, sum(n_elec), with_r=quantum_nuclei))
    raw_sampler = build_sampler(
        log_prob_fn,
        conf_init_fn,
        name=sample_cfg.sampler,
        adaptive=sample_cfg.get("adaptive", None),
        **sample_cfg.get(sample_cfg.sampler, {}))
    sampler = make_multistep(raw_sampler, n_step=n_multistep, concat=False)
    sampler = make_batched(sampler, n_batch=n_batch, concat=True)
    sampler = jax.tree_map(PAXIS.pmap if multi_device else jax.jit, sampler)

    # make optimizer
    lr_schedule = (optimize_cfg.lr if callable(optimize_cfg.lr) else
                   build_lr_schedule(**optimize_cfg.lr))
    optimizer = build_optimizer(
        loss_and_grad,
        name=optimize_cfg.optimizer,
        lr_schedule=lr_schedule,
        value_func_has_aux=True,
        value_func_has_rng=False,
        value_func_has_state=False,
        multi_device=multi_device,
        pmap_axis_name=PAXIS.name,
        log_psi_func=log_psi_fn,
        grad_clipping=optimize_cfg.get("grad_clipping", None),
        **optimize_cfg.get(optimize_cfg.optimizer, {}))

    # make training states
    if "states" in restart_cfg and restart_cfg.states:
        if isinstance(restart_cfg.states, str):
            LOGGER.info("Loading parameters and states from saved file")
            state_path = multi_process_name(restart_cfg.states)
            train_state = TrainingState(*load_pickle(state_path))
        else:
            LOGGER.info("Restart from parameters and states in config")
            train_state = TrainingState(*restart_cfg.states)
        train_state = match_loaded_state_to_device(train_state, multi_device)
    else:
        LOGGER.info("Initializing parameters and states")
        assert key is not None, \
            "key is required if not restarting from previous state"
        # initialize params
        if "params" in restart_cfg and restart_cfg.params:
            if isinstance(restart_cfg.params, str):
                LOGGER.info("Loading parameters from saved file")
                param_path = multi_process_name(restart_cfg.params)
                params = load_pickle(param_path)
            else:
                LOGGER.info("Restart from parameters in config")
                params = restart_cfg.params
            if isinstance(params, tuple):
                params = params[1]
            if isinstance(params, ConfigDict): # happen to be converted
                params = params.to_dict()
        else:
            key, parkey, skey = jax.random.split(key, 3) # init use single key
            fake_input = conf_init_fn(skey)
            params = (ansatz.init(parkey, *fake_input) if quantum_nuclei
                      else ansatz.init(parkey, fake_input))
        if multi_device:
            params = kfac_jax.utils.replicate_all_local_devices(params)
        # after init params, use pmapped key
        if multi_device:
            key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
        key, mckey, optkey, statekey = rng_split(key, 4)
        # initialize mc state
        if "chains" in restart_cfg and restart_cfg.chains:
            if isinstance(restart_cfg.chains, str):
                LOGGER.info("Loading MC states from saved file")
                chains_path = multi_process_name(restart_cfg.chains)
                mc_state = load_pickle(chains_path)
            else:
                LOGGER.info("Restart from MC states in config")
                mc_state = restart_cfg.chains
            _ref_shape = jax.eval_shape(sampler.init, mckey, params)[0].shape
            if len(mc_state) == 4 and mc_state[0].shape != _ref_shape:
                mc_state = mc_state[2] # assuming TrainingState
            mc_state = sampler.refresh(mc_state, params)
        else:
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
        return TrainingState(key, params, mc_state, opt_state), (mc_info, opt_info), data

    return training_step


def run(step_fn, train_state, iterations, log_cfg):
    """run the optimization loop (sample + update)"""

    log_cfg = ConfigDict(log_cfg)

    use_tensorboard = log_cfg.get("use_tensorboard", False)
    if use_tensorboard:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            LOGGER.warning("no tensorboardx, set use_tensorboard = False")
            use_tensorboard = False

    # writer and tensorboard tracker
    if jax.process_index() == 0:
        backup_if_exist(log_cfg.stat_path, prefix="bck",
                        max_keep=log_cfg.get("stat_keep"))
        writer = open(log_cfg.stat_path, "w", buffering=1)
        if use_tensorboard:
            tracker = SummaryWriter(log_cfg.tracker_path)
    # training log printer
    print_fields = {"step": "",
                    "e_tot": ".4f", "std_e": ".3e",
                    "acc": ".2f", "hacc": ".2f", "lr": ".2e"}
    printer = Printer(print_fields, time_format=".2f")
    # sample trajectory dumper
    dumper = None
    if log_cfg.dump_every > 0 and log_cfg.dump_path:
        from npy_append_array import NpyAppendArray
        dump_path = multi_process_name(log_cfg.dump_path)
        backup_if_exist(dump_path, prefix="bck",
                        max_keep=log_cfg.get("dump_keep"))
        assert not os.path.exists(dump_path)
        dumper = NpyAppendArray(dump_path, delete_if_exists=True)
    # determine chekcpoint path for potential multi device
    ckpt_path = multi_process_name(log_cfg.ckpt_path)

    # mysterious step to prevent kfac memory error
    train_state = jax.tree_map(jnp.copy, train_state)

    LOGGER.info("Start training")
    if jax.process_index() == 0:
        printer.print_header("# ")

    for ii in range(iterations):
        printer.reset_timer()

        # main training loop
        train_state, (mc_info, opt_info), sample_data = step_fn(train_state)

        # nan check
        if not jax.tree_util.tree_all(
          jax.tree_map(lambda a: jnp.all(~jnp.isnan(a)), train_state.params)):
            raise ValueError(f"NaN found in params at step {ii} "
                             f"(log step {int(opt_info['step'].mean())-1})")
        if jnp.any(opt_info["aux"]["nans"] > 0):
            LOGGER.warning("%d NaN(s) found in local energy at step %d (log step %d)",
                           opt_info["aux"]["nans"].sum(),
                           ii, opt_info['step'].mean()-1)

        # log stats
        if ((ii % log_cfg.stat_every == 0 or ii == iterations-1)
          and jax.process_index() == 0): # only print for process 0 (all same)
            acc_rate = PAXIS.all_mean(mc_info["is_accepted"])
            hacc_rate = 1. / PAXIS.all_mean(mc_info["recip_ratio"])
            ostep = opt_info["step"]
            istep = ostep - 1 if jnp.max(ostep) >= 0 else ii
            stat_dict = {"step": istep, **opt_info["aux"],
                         "acc": acc_rate, "hacc": hacc_rate,
                         "lr":opt_info["learning_rate"]}
            if "tuned_hparam" in mc_info:
                stat_dict["mc_hp"] = PAXIS.all_mean(mc_info["tuned_hparam"])
            stat_dict = {k: v[0] if jnp.ndim(v) > 0 else v # collect from potential pmap
                         for k, v in stat_dict.items()}
            printer.print_fields(stat_dict)
            onp.savetxt(writer, onp.array(list(stat_dict.values())).reshape(1,-1),
                        header=" ".join(stat_dict.keys()) if ii == 0 else "")
            if use_tensorboard:
                for k,v in stat_dict.items():
                    if k != "step":
                        tracker.add_scalar(k, v, stat_dict["step"])

        # dump traj
        if dumper is not None and ii % log_cfg.dump_every == 0:
            sconf, slogw = sample_data
            flat_conf = onp.concatenate([
                onp.asarray(sc).reshape(slogw.size, -1)
                for sc in jax.tree_util.tree_leaves(sconf)
            ], axis=-1)[None]
            dumper.append(flat_conf) # [n_step, n_batch, n_coord]

        # checkpoint
        if ii % log_cfg.ckpt_every == 0 or ii == iterations-1:
            save_checkpoint(
                ckpt_path,
                tuple(trim_training_state(train_state)),
                max_keep=log_cfg.get("ckpt_keep"))

    if jax.process_index() == 0:
        writer.close()
        if use_tensorboard:
            tracker.close()
    if dumper is not None:
        dumper.close()

    return train_state


def main(cfg):
    cfg = ConfigDict(cfg)

    import subprocess as sp
    try:
        cfg._git_hash = sp.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except sp.CalledProcessError:
        cfg._git_hash = None

    if "hpar_path" in cfg.log and jax.process_index() == 0:
        with open(cfg.log.hpar_path, "w") as hpfile:
            print(cfg_to_yaml(cfg), file=hpfile)

    import logging
    verbosity = getattr(logging, cfg.verbosity.upper())
    LOGGER.setLevel(verbosity)

    key = jax.random.PRNGKey(cfg.seed) if 'seed' in cfg else None
    system, ansatz, loss_fn, sampler, optimizer, train_state \
        = prepare(cfg.system, cfg.ansatz, cfg.sample, cfg.loss, cfg.optimize,
                  cfg.quantum_nuclei, key, cfg.restart, cfg.multi_device)

    training_step = build_training_step(sampler, optimizer)
    train_state = run(training_step, train_state, cfg.optimize.iterations, cfg.log)

    return train_state
