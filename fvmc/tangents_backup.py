import os

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.flatten_util import ravel_pytree
from ml_collections import ConfigDict

from . import LOGGER
from .wavefunction import log_prob_from_model, log_psi_from_model
from .utils import (adaptive_grad, symmetrize,
                    load_pickle, backup_if_exist, Printer)
from .sampler import build_conf_init_fn, build_sampler, make_batched
from .estimator import build_eval_local_elec
from .train import (parse_system_cfg, parse_ansatz_cfg, save_cfg_to_yaml,
                    load_ansatz_params, load_sampler_state,
                    multi_process_name, match_loaded_state_to_device)


def build_eval_tangents(eval_local_fn):
    """Build a function to evaluate the tangents of wavefunction and local energy."""

    def _log_psi_fn(params, conf):
        _, sign, logf, _ = eval_local_fn(params, conf)
        if jnp.iscomplexobj(sign):
            logf += jnp.log(sign)
        return logf

    def _e_local_fn(params, conf):
        e_loc, _, _, _ = eval_local_fn(params, conf)
        return e_loc

    def eval_tangents(params, data):
        """Evaluate the tangents of wavefunction and local energy."""
        # partial functions
        local_fn = lambda x: eval_local_fn(params, x)
        d_logf_fn = lambda x: ravel_pytree(adaptive_grad(_log_psi_fn, 0)(params, x))[0]
        d_eloc_fn = lambda x: ravel_pytree(adaptive_grad(_e_local_fn, 0)(params, x))[0]
        # evaluate with batch
        conf, logsw = data
        eloc, sign, logf, extras = jax.vmap(local_fn, 0, 0)(conf)
        eloc = eloc.reshape(-1, 1) # add last axis for later use
        d_logf = jax.vmap(d_logf_fn, 0, 0)(conf) # [n_samples, n_params]
        d_eloc = jax.vmap(d_eloc_fn, 0, 0)(conf) # [n_samples, n_params]
        # build matrix components, s_half and h_half are [n_samples, n_params + 1]
        # |psi_v> = d |psi_0> / d v for v > 0 and |psi_v> = |psi_0> for v = 0
        # <x|psi_v> / <x|psi_0> (= d log psi_0 / d v for v > 0)
        s_half = jnp.concatenate([jnp.ones_like(eloc), d_logf], axis=-1)
        # <x|H|psi_v> / <x|psi_0> (= d E_l / d v + E_l * d log psi_0 / d v for v > 0)
        h_half = jnp.concatenate([eloc, d_eloc + eloc * d_logf], axis=-1)
        # log of ratio between desired weight and sample weight
        log_weight = 2 * logf.real - logsw
        # return tangents and weights
        return h_half, s_half, log_weight

    return eval_tangents


@jax.jit
def assemble_matrices(h_half, s_half, log_weight):
    """Assemble hamiltonian and overlap matrices in tangent space."""
    weight = jnp.exp(log_weight)
    h_mat = jnp.einsum('ni,nj,n->ij', s_half.conj(), h_half, weight)
    s_mat = jnp.einsum('ni,nj,n->ij', s_half.conj(), s_half, weight)
    return symmetrize(h_mat), symmetrize(s_mat)


def build_sample_weight_fn(ansatz):
    log_psi_fn = log_psi_from_model(ansatz)
    log_prob_fn = log_prob_from_model(ansatz)
    # log (sum_v |d_v log psi|^2) + log |psi|^2
    def log_weight_fn(params, conf):
        dlogpsi = adaptive_grad(log_psi_fn, 0)(params, conf)
        wratio = jtu.tree_reduce(lambda a, b: a + (b * b.conj()).real.sum(), dlogpsi, 0)
        return jnp.log(wratio) + log_prob_fn(params, conf)
    return log_weight_fn


def prepare(system_cfg, ansatz_cfg, sample_cfg, eval_cfg, restart_cfg, key=None):
    if jax.device_count() > 1:
        LOGGER.warning("Tangents calculation does not support multi-device mode."
                       "Found %d devices. Only 1 will be used.", jax.device_count())

    # parse system and ansatz cfg
    system = parse_system_cfg(system_cfg)
    n_elec, elems, nuclei, cell = system
    ansatz = parse_ansatz_cfg(ansatz_cfg, system, quantum_nuclei=False)

    # build sampling log prob function
    reweighting = (eval_cfg.get("reweighting", False) or
                   sample_cfg.get("reweighting", False))
    log_prob_fn = (build_sample_weight_fn(ansatz) if reweighting
                   else log_prob_from_model(ansatz))

    # build sampler
    sample_cfg = ConfigDict(sample_cfg)
    n_batch = sample_cfg.size
    if sample_cfg.get("chains", None):
        LOGGER.warning("Ignoring `chains` in sampler config. Using %d chains.", n_batch)
    conf_init_fn = sample_cfg.get("conf_init_fn", build_conf_init_fn(
        elems, nuclei, sum(n_elec), with_r=False))
    sampler = build_sampler(
        log_prob_fn,
        conf_init_fn,
        name=sample_cfg.sampler,
        adaptive=sample_cfg.get("adaptive", None), # prevent bias
        **sample_cfg.get(sample_cfg.sampler, {}))
    sampler = make_batched(sampler, n_batch=n_batch, concat=False)
    sampler = jtu.tree_map(jax.jit, sampler)

    # build eval tangents function
    ke_kwargs = eval_cfg.get("ke_kwargs", {})
    pe_kwargs = eval_cfg.get("pe_kwargs", {})
    ext_pots = system_cfg.get("external_potentials", {})
    spin_pots = system_cfg.get("spin_potentials", {})
    lclargs = dict(ke_kwargs=ke_kwargs, pe_kwargs=pe_kwargs,
                   ext_pots=ext_pots, spin_pots=spin_pots,
                   stop_gradient=False)
    local_fn = build_eval_local_elec(ansatz, elems, nuclei, cell, **lclargs)
    eval_fn = jax.jit(build_eval_tangents(local_fn))

    # load ansatz params and sampler state
    if "states" in restart_cfg and restart_cfg.states:
        if isinstance(restart_cfg.states, str):
            LOGGER.info("Loading parameters and states from saved file")
            state_path = multi_process_name(restart_cfg.states)
            train_state = load_pickle(state_path)
        else:
            LOGGER.info("Restart from parameters and states in config")
            train_state = restart_cfg.states
        train_state = match_loaded_state_to_device(train_state, False)
        key, params, mc_state, *_ = train_state
    else:
        assert "params" in restart_cfg and restart_cfg.params, \
            "params is required if not restarting from previous state"
        # load ansatz params
        params = load_ansatz_params(restart_cfg.params)
        # load or initialize mc state
        if "chains" in restart_cfg and restart_cfg.chains:
            mc_state = load_sampler_state(restart_cfg.chains)
            mc_state = sampler.refresh(mc_state, params)
        else:
            assert key is not None, \
            "key is required if not restarting from previous mc state"
            key, mckey = jax.random.split(key)
            mc_state = sampler.init(mckey, params)
        if "burn_in" in sample_cfg and sample_cfg.burn_in > 0:
            LOGGER.info(f"Burning in the sampler for {sample_cfg.burn_in} steps")
            key, subkey = jax.random.split(key)
            mc_state = sampler.burn_in(subkey, params, mc_state, sample_cfg.burn_in)

    return system, ansatz, sampler, eval_fn, (key, params, mc_state)


def run(sampler, eval_fn, state, iterations,
        save_folder, save_mode='auto', save_every=1):
    # parse save mode
    save_mode = save_mode.lower()
    if not (save_mode in ['auto', 'dense', 'sparse']
            or save_mode.startswith('acc')):
        raise ValueError(f"Invalid save_mode: {save_mode}")
    # printing
    print_fields = {"step": "", "acc": ".2f", "hacc": ".2f"}
    printer = Printer(print_fields, time_format=".2f")

    h_mat = s_mat = 0. # only for acc mode
    key, params, mc_state = state # unpack state

    printer.print_header("# ")
    for i in range(iterations):
        printer.reset_timer()

        # main calculation
        key, subkey = jax.random.split(key)
        mc_state, data, mc_info = sampler.sample(subkey, params, mc_state)
        h_half, s_half, log_w = eval_fn(params, data)

        # save results
        if save_mode.startswith('acc'): # save accumulated matrix, save disc space
            if save_mode.endswith("cpu"):
                h_half, s_half, log_w = jax.device_put((h_half, s_half, log_w),
                                                       jax.devices('cpu')[0])
            h_new, s_new = assemble_matrices(h_half, s_half, log_w)
            h_mat += (h_new - h_mat) / (i + 1)
            s_mat += (s_new - s_mat) / (i + 1)
            if (i+1) % save_every == 0 or i == iterations - 1:
                backup_if_exist(f"{save_folder}/h_mat.npy", max_keep=2, prefix="last")
                backup_if_exist(f"{save_folder}/s_mat.npy", max_keep=2, prefix="last")
                jnp.save(f"{save_folder}/h_mat.npy", h_mat)
                jnp.save(f"{save_folder}/s_mat.npy", s_mat)
        else: # save every step into separate folder
            if save_every != 1:
                LOGGER.warning("save_every is ignored when save_mode is not 'acc'")
            basepath = f"{save_folder}/step.{i:0{len(str(iterations))}d}"
            os.mkdir(basepath)
            # save sampled configuration
            sconf, slogw = data
            flat_conf = jnp.concatenate([
                sc.reshape(slogw.size, -1) for sc in jtu.tree_leaves(sconf)
            ], axis=-1)
            jnp.save(f"{basepath}/conf.npy", flat_conf)
            jnp.save(f"{basepath}/log_weight.npy", log_w)
            # save matrices
            n_sample, n_param = h_half.shape
            dense_mode = (save_mode == 'dense' or
                        (save_mode == 'auto' and n_param < n_sample))
            if dense_mode:
                h_mat, s_mat = assemble_matrices(h_half, s_half, log_w)
                jnp.save(f"{basepath}/h_mat.npy", h_mat)
                jnp.save(f"{basepath}/s_mat.npy", s_mat)
            else:
                jnp.save(f"{basepath}/h_half.npy", h_half)
                jnp.save(f"{basepath}/s_half.npy", s_half)

        acc_rate = jnp.mean(mc_info["is_accepted"]).item()
        hacc_rate = 1. / jnp.mean(mc_info["recip_ratio"]).item()
        printer.print_fields({"step": i, "acc": acc_rate, "hacc": hacc_rate})


def main(cfg):
    cfg = ConfigDict(cfg)
    eval_cfg = cfg.eval_tangents

    import logging
    verbosity = getattr(logging, cfg.verbosity.upper())
    LOGGER.setLevel(verbosity)

    # backup existing folder
    save_folder = eval_cfg.save_folder.rstrip("/")
    if os.path.exists(save_folder) and os.path.samefile(save_folder, "."):
        save_folder = "tangents"
    backup_if_exist(save_folder, prefix="bck")
    os.makedirs(save_folder, exist_ok=False)
    # save config as yaml
    save_cfg_to_yaml(cfg, f"{save_folder}/hparams.yaml")

    key = jax.random.PRNGKey(cfg.seed) if 'seed' in cfg else None
    system, ansatz, sampler, eval_fn, state \
        = prepare(cfg.system, cfg.ansatz, cfg.sample, eval_cfg, cfg.restart, key)

    run(sampler, eval_fn, state, eval_cfg.iterations, save_folder,
        eval_cfg.get('save_mode', 'auto'), eval_cfg.get('save_every', 1))