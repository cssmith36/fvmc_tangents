import os

from functools import partial

import jax
import flax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.flatten_util import ravel_pytree
from ml_collections import ConfigDict

from jax.experimental.host_callback import id_print

from . import LOGGER
from .wavefunction import log_prob_from_model, log_psi_from_model, log_psi_from_frozen_model
from .wavefunction.base import FrozenModelD2, FrozenModelD3
from .utils import (adaptive_grad, adaptive_grad_fwd, adaptive_hessian, symmetrize,
                    load_pickle, backup_if_exist, Printer, Array)
from .sampler import build_conf_init_fn, build_sampler, build_tangent_sampler, make_tangent_batched,  make_batched
from .estimator import build_eval_local_elec
from .train import (parse_system_cfg, parse_ansatz_cfg, save_cfg_to_yaml,
                    load_ansatz_params, load_sampler_state,
                    multi_process_name, match_loaded_state_to_device)
import functools
import h5py

import shutil
import os
from typing import Callable, Sequence, Literal, Tuple, Any

import flax.linen as nn

def einshape(arr):
    return np.array([i for i in range(len(arr.shape))])

def _wrap_sign(sign, logp):
    if jnp.iscomplexobj(sign):
        logp += jnp.log(sign)
    return logp

def gen_calc_dens(cell: Array, spins: Sequence[int], bins: Sequence[int]):
    return partial(calc_dens, cell=cell, spins=spins, bins=bins)

@partial(jax.jit, static_argnames=('bins', 'spins'))
def calc_dens(walker: Array, weights: Array, cell: Array, spins: Sequence[int], bins: Sequence[int]):
    # normalized so that the integral of rho is the number of particles
    # numerically, sum(hist * bin_vol) = n_elec
    nsample, nelec, ndim = walker.shape
    hrange = [(0, 1)] * ndim
    bin_vol = 1.#jnp.prod(jnp.diff(hrange, axis=-1)) / jnp.prod(bins)
    #split_idx = jnp.cumsum((spins))[:-1]
    invvec = jnp.linalg.inv(cell)
    frac_walker = (walker @ invvec) % 1

    res = []
    ### Test Vectorless Version
    #weights = jnp.insert(jnp.zeros(nsample-1),0,1.)
    hist_fn = lambda x,w: w*jnp.histogramdd(x, bins=bins, range=hrange)[0]
    return jnp.sum(jax.vmap(hist_fn,in_axes=(0,0))(frac_walker,weights[...,None]),axis=0)
    #return res#, dict(edges=edges)
    
def build_eval_observables(eval_local_fn,dev_2,eval_local_fn_frozen=None,nn_firstdev=False):
    def _log_psi_fn(params,p_junk,conf):
        _, sign, logf, _ = eval_local_fn(params, conf)
        if jnp.iscomplexobj(sign):
            logf += jnp.log(sign)
        return logf

    def _log_psi_fn_frozen(frozen_params,backflow_params, conf):
        _, sign, logf, _ = eval_local_fn_frozen(frozen_params, conf, backflow_params=backflow_params)
        if jnp.iscomplexobj(sign):
            logf += jnp.log(sign)
        return logf

    def eval_observable(params, data, params_split=None):
        """Evaluate the tangents of wavefunction and local energy."""
        # partial functions
        local_fn = lambda x: eval_local_fn(params, x)
        
        if dev_2:
            if eval_local_fn_frozen:
                log_psi_fun = _log_psi_fn_frozen
                
            else:
                log_psi_fun = _log_psi_fn
                params_split = {'frozen_params':params,'backflow_params':None}

        
            d_logf_trunc_fn = lambda x: adaptive_grad(log_psi_fun, 0)(params_split['frozen_params'],params_split['backflow_params'], x)

            
            conf, logsw = data
            eloc, sign, logf, extras = jax.vmap(local_fn, 0, 0)(conf)
            eloc = eloc.reshape(-1, 1) # add last axis for later use

            d2_logf_fn = lambda x: adaptive_hessian(log_psi_fun, 0)(params_split['frozen_params'],params_split['backflow_params'],x)

            ### Calculate Second Derivatives
            d_logf_trunc = jax.vmap(d_logf_trunc_fn, 0, 0)(conf)
            d_fg_trunc = jax.vmap(lambda A,B: jax.tree_map(lambda B: jax.tree_map(lambda A: jnp.einsum(B,einshape(B),A,einshape(A)+len(einshape(B)),np.concatenate([einshape(B),einshape(A)+len(einshape(B))])),A),B))(d_logf_trunc,d_logf_trunc)

            d2_logf_trunc = jax.vmap(d2_logf_fn, in_axes=0, out_axes=0)(conf)

            d2_f = jax.tree_map(lambda x,y: x+y,d2_logf_trunc,d_fg_trunc)
            d2_f_r = jax.vmap(lambda x: ravel_pytree(x)[0])(d2_f)

            ### Construct dH for full first dv or just SJ
            if nn_firstdev:
                d_logf_fn = lambda x: adaptive_grad(_log_psi_fn, 0)(params_split['frozen_params'],params_split['backflow_params'], x)
                d_logf = jax.vmap(d_logf_fn, 0, 0)(conf)
                d_logf_r = jax.vmap(lambda x: ravel_pytree(x)[0])(d_logf)

            else:
                d_logf_r = jax.vmap(lambda x: ravel_pytree(x)[0])(d_logf_trunc)


            # build matrix components, s_half and h_half are [n_samples, n_params + 1]

            s_half = jnp.concatenate([jnp.ones_like(eloc), d_logf_r, d2_f_r], axis=-1)
            print('s half shape',s_half.shape)
            
        else:
            d_logf_fn = lambda x: ravel_pytree(adaptive_grad(_log_psi_fn, 0)(params, None, x))[0]
            #d_eloc_fn = lambda x: ravel_pytree(adaptive_grad(_e_local_fn, 0)(params, x))[0]
            conf, logsw = data
            eloc, sign, logf, extras = jax.vmap(local_fn, 0, 0)(conf)
            eloc = eloc.reshape(-1, 1) # add last axis for later use
            d_logf = jax.vmap(d_logf_fn, 0, 0)(conf) # [n_samples, n_params]
            s_half = jnp.concatenate([jnp.ones_like(eloc), d_logf], axis=-1)
            print('s half shape:',s_half.shape)
            # <x|H|psi_v> / <x|psi_0> (= d E_l / d v + E_l * d log psi_0 / d v for v > 0)
        # log of ratio between desired weight and sample weight
        log_weight = 2 * logf.real - logsw


        # return tangents and weights
        return s_half, log_weight, jnp.mean(eloc)# eloc, d2_logf, d2_eloc, d_logf, d_eloc

    return eval_observable
        

def build_eval_tangents(eval_local_fn,dev_2,eval_local_fn_frozen=None,nn_firstdev=False):
    """Build a function to evaluate the tangents of wavefunction and local energy."""

    def _log_psi_fn(params,p_junk,conf):
        _, sign, logf, _ = eval_local_fn(params, conf)
        if jnp.iscomplexobj(sign):
            logf += jnp.log(sign)
        return logf

    def _e_local_fn(params,p_junk,conf):
        e_loc, _, _, _ = eval_local_fn(params, conf)
        return e_loc

    def _log_psi_fn_frozen(frozen_params,backflow_params, conf):
        _, sign, logf, _ = eval_local_fn_frozen(frozen_params, conf, backflow_params=backflow_params)
        if jnp.iscomplexobj(sign):
            logf += jnp.log(sign)
        return logf

    def _e_local_fn_frozen(frozen_params,backflow_params,conf):
        e_loc, _, _, _ = eval_local_fn_frozen(frozen_params, conf, backflow_params=backflow_params)
        return e_loc
        

    def eval_tangents(params, data, params_split=None):
        """Evaluate the tangents of wavefunction and local energy."""
        # partial functions
        local_fn = lambda x: eval_local_fn(params, x)
        
        if dev_2:
            if eval_local_fn_frozen:
                log_psi_fun = _log_psi_fn_frozen
                eval_local_fun = _e_local_fn_frozen
                
            else:
                log_psi_fun = _log_psi_fn
                eval_local_fun = _e_local_fn
                params_split = {'frozen_params':params,'backflow_params':None}

            d_logf_trunc_fn = lambda x: adaptive_grad(log_psi_fun, 0)(params_split['frozen_params'],params_split['backflow_params'], x)
            d_eloc_trunc_fn = lambda x: adaptive_grad(eval_local_fun, 0)(params_split['frozen_params'],params_split['backflow_params'], x)
            
            conf, logsw = data
            eloc, sign, logf, extras = jax.vmap(local_fn, 0, 0)(conf)
            eloc = eloc.reshape(-1, 1) # add last axis for later use

            d2_logf_fn = lambda x: adaptive_hessian(log_psi_fun, 0)(params_split['frozen_params'],params_split['backflow_params'],x)
            d2_eloc_fn = lambda x: adaptive_hessian(eval_local_fun, 0)(params_split['frozen_params'],params_split['backflow_params'],x)

            ### Calculate Second Derivatives
            d_logf_trunc = jax.vmap(d_logf_trunc_fn, 0, 0)(conf)
            d_fg_trunc = jax.vmap(lambda A,B: jax.tree_map(lambda B: jax.tree_map(lambda A: jnp.einsum(B,einshape(B),A,einshape(A)+len(einshape(B)),np.concatenate([einshape(B),einshape(A)+len(einshape(B))])),A),B))(d_logf_trunc,d_logf_trunc)

            d_eloc_trunc = jax.vmap(d_eloc_trunc_fn, 0, 0)(conf)
        
            d2_logf_trunc = jax.vmap(d2_logf_fn, in_axes=0, out_axes=0)(conf)
            d2_eloc_trunc = jax.vmap(d2_eloc_fn, in_axes=0, out_axes=0)(conf)

            d2_f = jax.tree_map(lambda x,y: x+y,d2_logf_trunc,d_fg_trunc)
            d2_f_r = jax.vmap(lambda x: ravel_pytree(x)[0])(d2_f)

            ### Construct dH
            dH_trunc = jax.vmap(lambda e,y,z: jax.tree_map(lambda x,w: e*x + w,y,z), in_axes=(0,0,0))(eloc,d_logf_trunc,d_eloc_trunc)

            ### Construct dH for full first dv or just SJ
            if nn_firstdev:
                d_logf_fn = lambda x: adaptive_grad(_log_psi_fn, 0)(params,None,x)#(params_split['frozen_params'],params_split['backflow_params'], x)
                d_logf = jax.vmap(d_logf_fn, 0, 0)(conf)
                d_logf_r = jax.vmap(lambda x: ravel_pytree(x)[0])(d_logf)

                d_eloc_fn = lambda x: adaptive_grad(_e_local_fn, 0)(params,None,x)#(params_split['frozen_params'],params_split['backflow_params'], x)
                d_eloc = jax.vmap(d_eloc_fn, 0, 0)(conf)

                dH = jax.vmap(lambda e,y,z: jax.tree_map(lambda x,w: e*x + w,y,z), in_axes=(0,0,0) )(eloc,d_logf,d_eloc)
                dH_r = jax.vmap(lambda x: ravel_pytree(x)[0])(dH)

            else:
                dH_r = jax.vmap(lambda x: ravel_pytree(x)[0])(dH_trunc)
                d_logf_r = jax.vmap(lambda x: ravel_pytree(x)[0])(d_logf_trunc)
            ### Construct h2_half
            trm1 = lambda A,B: ravel_pytree(jax.tree_map(lambda B: jax.tree_map(lambda A: jnp.einsum(B,einshape(B),A,einshape(A)+len(einshape(B)), np.concatenate([einshape(B),einshape(A)+len(einshape(B))])),A),B))[0]
            trm1 = jax.vmap(trm1)(d_eloc_trunc,d_logf_trunc)
            trm2 = lambda e,x: ravel_pytree(jax.tree_map(lambda x: e*x,x))[0]
            trm2 = jax.vmap(trm2)(eloc,d2_logf_trunc)

            trm3 = jax.vmap(lambda x: ravel_pytree(x)[0])(d2_eloc_trunc)
            trm4 = lambda A,B: ravel_pytree(jax.tree_map(lambda B: jax.tree_map(lambda A: jnp.einsum(B,einshape(B),A,einshape(A)+len(einshape(B)), np.concatenate([einshape(B),einshape(A)+len(einshape(B))])),A),B))[0]
            trm4 = jax.vmap(trm4)(d_logf_trunc,dH_trunc)
            
            h2_half = trm1 + trm2 + trm3 + trm4

            # build matrix components, s_half and h_half are [n_samples, n_params + 1]
            s_half = jnp.concatenate([jnp.ones_like(eloc), d_logf_r, d2_f_r], axis=-1)
            print('s half shape',s_half.shape)
            h_half = jnp.concatenate([eloc, dH_r, h2_half], axis=-1)
            print('h half shape',h_half.shape)
            
        else:
            d_logf_fn = lambda x: ravel_pytree(adaptive_grad(_log_psi_fn, 0)(params,None, x))[0]
            d_eloc_fn = lambda x: ravel_pytree(adaptive_grad(_e_local_fn, 0)(params,None, x))[0]
            conf, logsw = data
            eloc, sign, logf, extras = jax.vmap(local_fn, 0, 0)(conf)
            eloc = eloc.reshape(-1, 1) # add last axis for later use
            d_logf = jax.vmap(d_logf_fn, 0, 0)(conf) # [n_samples, n_params]
            d_eloc = jax.vmap(d_eloc_fn, 0, 0)(conf) # [n_samples, n_params]
            s_half = jnp.concatenate([jnp.ones_like(eloc), d_logf], axis=-1)
            # <x|H|psi_v> / <x|psi_0> (= d E_l / d v + E_l * d log psi_0 / d v for v > 0)
            h_half = jnp.concatenate([eloc, d_eloc + eloc * d_logf], axis=-1)
        # log of ratio between desired weight and sample weight
        log_weight = 2 * logf.real - logsw


        # return tangents and weights
        return h_half, s_half, log_weight#, jnp.mean(eloc)#, eloc, d2_logf, d2_eloc, d_logf, d_eloc

    return eval_tangents

@jax.jit
def assemble_h_matrix(h_half, s_half, log_weight):
    """Assemble hamiltonian and overlap matrices in tangent space."""
    weight = jnp.exp(log_weight)
    print('assembling h...')
    h_mat = jnp.einsum('ni,nj,n->ij', s_half.conj(), h_half, weight)
    print('symmetrizing...')
    return symmetrize(h_mat)
@jax.jit
def assemble_s_matrix(s_half, log_weight):
    """Assemble hamiltonian and overlap matrices in tangent space."""
    weight = jnp.exp(log_weight)
    print('assembling s...')
    s_mat = jnp.einsum('ni,nj,n->ij', s_half.conj(), s_half, weight)
    print('symmetrizing...')
    return symmetrize(s_mat)


def build_sample_weight_fn(ansatz,dev_2,frozen_ansatz=None,params_split=None,sample_nn=False):
    log_psi_fn = log_psi_from_model(ansatz)
    if params_split:
        log_psi_fn_frozen = log_psi_from_frozen_model(frozen_ansatz)
    log_prob_fn = log_prob_from_model(ansatz)
    # log (sum_v |d_v log psi|^2) + log |psi|^2
    #def log_weight_fn(params,weights,conf):
    def log_weight_fn(params,conf):
        weights = 1.
        sample_dev_2 = True
        if params_split and sample_dev_2:
            logpsifun = log_psi_fn_frozen
            dlogpsi = adaptive_grad(logpsifun, 0)(params_split['frozen_params'], params_split['backflow_params'], conf)
            if sample_nn:
                dlogpsi_full = adaptive_grad(log_psi_fn, 0)(params, conf)
        else:
            logpsifun = log_psi_fn
            dlogpsi = adaptive_grad(logpsifun, 0)(params, conf)

        ### handle sample_nn case tk
        if sample_nn:
            #wratio = jtu.tree_reduce(lambda a, b: a + (b * b.conj()).real.sum(), dlogpsi_full, 0)
            wratio = ravel_pytree(dlogpsi_full)[0]
        elif dev_2 and not sample_nn:
            wratio = ravel_pytree(dlogpsi)[0]
        elif not dev_2 and not sample_nn:
            wratio = ravel_pytree(dlogpsi)[0]

        if dev_2 and sample_dev_2:
            if params_split:
                d2logpsi = adaptive_hessian(logpsifun, 0)(params_split['frozen_params'], params_split['backflow_params'], conf)
            else:
                d2logpsi = adaptive_hessian(logpsifun, 0)(params, conf)

            d_fg = lambda A,B: jax.tree_map(lambda B: jax.tree_map(lambda A: jnp.einsum(B,einshape(B),A,einshape(A)+len(einshape(B)),np.concatenate([einshape(B),einshape(A)+len(einshape(B))])),A),B)
            d_fg = d_fg(dlogpsi,dlogpsi)
            d2_f = jax.tree_map(lambda x,y: x+y,d2logpsi,d_fg)
            d2_f = ravel_pytree(d2_f)[0]

            logprob = log_prob_fn(params, conf)
            full = jnp.concatenate([jnp.exp(logprob)[None,...],wratio,d2_f],axis=-1)
            print('sample shape:',full.shape)
            ### Should this really be the norm? (tk)
            full = (full*full.conj()*weights).real.sum()
            return jnp.log(full) + logprob
        else:
            logprob = log_prob_fn(params, conf)
            full = jnp.concatenate([jnp.exp(logprob)[None,...],wratio],axis=-1)
            return log_prob_fn(params, conf) #+ jnp.log((full*full.conj()).real.sum())
    return log_weight_fn

def prepare(system_cfg, ansatz_cfg, sample_cfg, eval_cfg, restart_cfg, key=None):
    if jax.device_count() > 1:
        LOGGER.warning("Tangents calculation does not support multi-device mode."
                       "Found %d devices. Only 1 will be used.", jax.device_count())

    # parse system and ansatz cfg
    system = parse_system_cfg(system_cfg)
    dev_2 = eval_cfg.get("dev_2",False)
    n_elec, elems, nuclei, cell = system
    ansatz = parse_ansatz_cfg(ansatz_cfg, system, quantum_nuclei=False)
    # build eval tangents function
    ke_kwargs = eval_cfg.get("ke_kwargs", {})
    pe_kwargs = eval_cfg.get("pe_kwargs", {})
    ext_pots = system_cfg.get("external_potentials", {})
    spin_pots = system_cfg.get("spin_potentials", {})
    lclargs = dict(ke_kwargs=ke_kwargs, pe_kwargs=pe_kwargs,
                   ext_pots=ext_pots, spin_pots=spin_pots,
                   stop_gradient=False)

    local_fn = build_eval_local_elec(ansatz, elems, nuclei, cell, **lclargs)
    eval_fn = None
    # load ansatz params
    if ansatz_cfg.freeze_params:
        params = load_ansatz_params(restart_cfg.params)
        
        if ansatz_cfg.split_params != None:
            params_split = ansatz_cfg.split_params
            frozen_ansatz = FrozenModelD3(ansatz,'backflow')
        else:
            frozen_params = flax.core.frozen_dict.copy(params)
            backflow_params = frozen_params['params'].pop('backflow')
            params_split = {'frozen_params':frozen_params,'backflow_params':backflow_params}
            frozen_ansatz = FrozenModelD2(ansatz,'backflow')
        
        local_fn_frozen = build_eval_local_elec(frozen_ansatz, elems, nuclei, cell, **lclargs)
        if eval_cfg.eval_obs:
            print('eval obs 1')
            eval_fn_frozen = jax.jit(build_eval_observables(local_fn,dev_2,local_fn_frozen))
        else:
            eval_fn_frozen = jax.jit(build_eval_tangents(local_fn,dev_2,local_fn_frozen))
    else:
        params = load_ansatz_params(restart_cfg.params)
        frozen_params = params
        backflow_params = None
        params_split = {'frozen_params':frozen_params,'backflow_params':backflow_params}
        frozen_ansatz = FrozenModelD2(ansatz,'backflow')
        eval_fn_frozen = None
        params_split = None
        if eval_cfg.eval_obs:
            print('eval obs full WF')
            eval_fn = jax.jit(build_eval_observables(local_fn,dev_2))
        else:
            eval_fn = jax.jit(build_eval_tangents(local_fn,dev_2))

    reweighting = (eval_cfg.get("reweighting", False) or
                   sample_cfg.get("reweighting", False))
    log_prob_fn = (build_sample_weight_fn(ansatz,dev_2,frozen_ansatz,params_split) if reweighting
                   else log_prob_from_model(ansatz))
    dev_2 = eval_cfg.get("dev_2", False)

    # build sampler
    sample_cfg = ConfigDict(sample_cfg)
    n_batch = sample_cfg.size
    if sample_cfg.get("chains", None):
        LOGGER.warning("Ignoring `chains` in sampler config. Using %d chains.", n_batch)
    conf_init_fn = sample_cfg.get("conf_init_fn", build_conf_init_fn(
        elems, nuclei, sum(n_elec), with_r=False))
    '''sampler = build_tangent_sampler(
        log_prob_fn,
        conf_init_fn,
        name=sample_cfg.sampler,
        adaptive=sample_cfg.get("adaptive", None), # prevent bias
        **sample_cfg.get(sample_cfg.sampler, {}))
    sampler = make_tangent_batched(sampler, n_batch=n_batch, concat=False)'''

    sampler = build_sampler(
        log_prob_fn,
        conf_init_fn,
        name=sample_cfg.sampler,
        adaptive=sample_cfg.get("adaptive", None), # prevent bias
        **sample_cfg.get(sample_cfg.sampler, {}))
    sampler = make_batched(sampler, n_batch=n_batch, concat=False)

    sampler = jtu.tree_map(jax.jit, sampler)

    # initialize sampler weights (handle sample_nn=True case eventually) (tk)
    if dev_2 and not params_split:
        tot_params = sum(x.size for x in jax.tree_leaves(params))
        basis_size = 1 + tot_params + tot_params**2

    elif dev_2 and params_split:
        tot_params = sum(x.size for x in jax.tree_leaves(params_split['frozen_params']))
        basis_size = 1 + tot_params + tot_params**2
    elif not dev_2 and not params_split:
        print('here!')
        basis_size = 1 + sum(x.size for x in jax.tree_leaves(params))
    sample_nn=False
    if sample_nn:
        tot_params = sum(x.size for x in jax.tree_leaves(params))
        dv2_params = sum(x.size for x in jax.tree_leaves(params_split['frozen_params']))
        basis_size = 1 + tot_params + dv2_params**2
    basis_size = 10
    weights = jnp.ones(basis_size)
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
            #mc_state = sampler.init(mckey, params, weights)
            mc_state = sampler.init(mckey, params)
        if "burn_in" in sample_cfg and sample_cfg.burn_in > 0:
            LOGGER.info(f"Burning in the sampler for {sample_cfg.burn_in} steps")
            key, subkey = jax.random.split(key)
            #mc_state = sampler.burn_in_tangent(subkey, params, weights, mc_state, sample_cfg.burn_in)
            mc_state = sampler.burn_in(subkey, params, mc_state, sample_cfg.burn_in)
    
    ### Load Frozen Model
    '''if ansatz_cfg.freeze_params:
        frozen_params = flax.core.frozen_dict.copy(params)
        backflow_params = frozen_params['params'].pop('backflow')
        params_split = {'frozen_params':frozen_params,'backflow_params':backflow_params}
        frozen_ansatz = FrozenModelD2(ansatz,'backflow')
        local_fn_frozen = build_eval_local_elec(frozen_ansatz, elems, nuclei, cell, **lclargs)
        eval_fn_frozen = jax.jit(build_eval_tangents(local_fn,dev_2,local_fn_frozen))
    else:
        frozen_ansatz = None
        eval_fn_frozen = None
        params_split = None'''

    return system, ansatz, frozen_ansatz, sampler, eval_fn, eval_fn_frozen, (key, params,params_split, mc_state)

@jax.jit
def assemble_matrices(h_half, s_half, log_weight):
    """Assemble hamiltonian and overlap matrices in tangent space."""
    weight = jnp.exp(log_weight)
    h_mat = jnp.einsum('ni,nj,n->ij', s_half.conj(), h_half, weight)
    s_mat = jnp.einsum('ni,nj,n->ij', s_half.conj(), s_half, weight)
    return symmetrize(h_mat), symmetrize(s_mat)

### tk
@jax.jit
def assemble_obs_matrices(s_half,log_weight, evecs):
    """Assemble hamiltonian and overlap matrices in tangent space."""
    weight = jnp.exp(log_weight)
    print('shapes',weight.shape,s_half.shape,evecs[1].shape,evecs[0].shape)
    obs_mat = jnp.einsum('ni,nj,n->n', evecs[1].conj()*s_half, evecs[0]*s_half.conj(), weight)
    #obs_mat = jnp.einsum('ni,nj->n', evecs[1]*s_half.conj(), evecs[0].conj()*s_half)
    return obs_mat.real

def run(sampler, eval_fn, eval_fn_frozen, state, iterations, save_folder, seeds, dense_mode=False, obs_args=None, evecs=None, dev_2=False):
    # unpack init state
    key, params, params_split, mc_state = state
    seed,save_seed = seeds
    # printing
    print_fields = {"step": "", "acc": ".2f", "hacc": ".2f"}
    printer = Printer(print_fields, time_format=".2f")
    #dev_2 = True
    # initialize sampler weights (handle sample_nn=True case eventually) (tk)
    if dev_2 and not params_split:
        tot_params = sum(x.size for x in jax.tree_leaves(params))
        basis_size = 1 + tot_params + tot_params**2
    elif dev_2 and params_split:
        print('Here!')
        tot_params = sum(x.size for x in jax.tree_leaves(params_split['frozen_params']))
        basis_size = 1 + tot_params + tot_params**2
    elif not dev_2:
        tot_params = sum(x.size for x in jax.tree_leaves(params))
        basis_size = 1 + tot_params
    else:
        raise Exception('first dv only not yet implemented')
    sample_nn = False
    if sample_nn:
        tot_params = sum(x.size for x in jax.tree_leaves(params))
        dv2_params = sum(x.size for x in jax.tree_leaves(params_split['frozen_params']))
        basis_size = 1 + tot_params + dv2_params**2
    basis_size = 10
    s_diag = jnp.ones(basis_size)

    printer.print_header("# ")
    n_digit = len(str(iterations - 1))
    cnt = 0
    obs_full=0.

    h_mat=s_mat=0. 

    for i in range(iterations):
        print('iter: ',i)
        printer.reset_timer()
        basepath = f"{save_folder}/steps/step.{i:0{n_digit}d}"
        basepath2 = f"{save_folder}"
        localpath = '/tmp'
        os.mkdir(basepath)

        key, subkey = jax.random.split(key)

        #mc_state, data, mc_info = sampler.sample(subkey, params, 1/jnp.sqrt(s_diag), mc_state)
        mc_state, data, mc_info = sampler.sample(subkey, params, mc_state)
        
        ### tk
        if evecs != None:
            #if not evecs:
            #    raise Exception("You chose eval_obs mode and did not specify the eigevectors \(~_~)/")
            if eval_fn_frozen:
                s_half,log_weight, eloc = eval_fn_frozen(params,data,params_split)
            else:
                s_half,log_weight, eloc = eval_fn(params,data)
            print('eloc:',eloc)

        else:
            if eval_fn_frozen:
                h_half, s_half, log_weight = eval_fn_frozen(params,data,params_split)
                print('here')

            else:
                h_half, s_half, log_weight = eval_fn(params, data)

        # for saving sampled configuration
        sconf, slogw = data
        flat_conf = jnp.concatenate([
            sc.reshape(slogw.size, -1) for sc in jtu.tree_leaves(sconf)
        ], axis=-1)
        save=True
        if save:

            #h_mat=s_mat=0. 
            if dense_mode:
                if i > 0:
                    if not evecs:
                        h_mat += assemble_h_matrix(h_half, s_half, log_weight)
                        #h_mat += np.load(localpath + '/h_mat_'+str(save_seed)+'.npy')
                        #jnp.save(localpath + '/h_mat_'+str(save_seed)+'.npy', h_mat)
                        #h_mat = 0.
                        s_mat += assemble_s_matrix(s_half, log_weight)
                        #s_mat += np.load(localpath + '/s_mat_'+str(save_seed)+'.npy')
                        #jnp.save(localpath + '/s_mat_'+str(save_seed)+'.npy', s_mat)
                        if i>50:
                            s_diag = jnp.diagonal(s_mat)/(2048*(i+1))
                        #s_mat = 0.
                    else:
                        #from fvmc.observable import gen_calc_dens
                        print('Here!',data[0].shape)
                        obs_weights = assemble_obs_matrices(s_half,log_weight,evecs)
                        #jnp.save('/mnt/home/csmith1/ceph/excitedStates/benchmark/n=5/k=3/reweight(2dev)/2048x1500steps/obs_dens_scratch_100_conj/evec=0/100Steps_vmap/obs/obs_weights.npy',obs_weights)
                        #jnp.save('/mnt/home/csmith1/ceph/excitedStates/benchmark/n=5/k=3/reweight(2dev)/2048x1500steps/obs_dens_scratch_100_conj/evec=0/100Steps_vmap/obs/log_weight.npy',log_weight)
                        #jnp.save('/mnt/home/csmith1/ceph/excitedStates/benchmark/n=5/k=3/reweight(2dev)/2048x1500steps/obs_dens_scratch_100_conj/evec=0/100Steps_vmap/obs/evecs.npy',evecs)
                        #jnp.save('/mnt/home/csmith1/ceph/excitedStates/benchmark/n=5/k=3/reweight(2dev)/2048x1500steps/obs_dens_scratch_100_conj/evec=0/100Steps_vmap/obs/s_half.npy',s_half)
                        calc_dens = gen_calc_dens(obs_args['cell'], obs_args['spins'], obs_args['bins'])
                        obs = calc_dens(data[0],obs_weights) 
                        obs_full += obs/(2048.*iterations)

                        #obs += jnp.load(localpath + '/obs_'+str(save_seed)+'.npy')
                        #jnp.save(localpath + '/obs_'+str(save_seed)+'.npy',obs)

                else:
                    if not evecs:
                        h_mat = assemble_h_matrix(h_half, s_half, log_weight)
                        #np.save(localpath + '/h_mat_'+str(save_seed)+'.npy', h_mat)
                        #h_mat = 0.
                        s_mat = assemble_s_matrix(s_half, log_weight)
                        #np.save(localpath + '/s_mat_'+str(save_seed)+'.npy', s_mat)
                        #s_mat = 0.
                        jnp.save(f'{basepath2}/h_mat_'+str(seed)+'_init.npy',h_mat)
                        jnp.save(f'{basepath2}/s_mat_'+str(seed)+'_init.npy',s_mat)
                    else:
                        #from fvmc.observable import gen_calc_dens
                        weights = assemble_obs_matrices(s_half, log_weight, evecs)
                        calc_dens = gen_calc_dens(obs_args['cell'], obs_args['spins'], obs_args['bins'])
                        obs = calc_dens(data[0],weights)
                        obs_full += obs/(2048*iterations)
                        #jnp.save(localpath + '/obs_'+str(save_seed)+'.npy',obs_full)
                        #obs_single = calc_dens(data[0])


                        #raise NotImplementedError

                np.save(f"{basepath2}/total_steps.npy", i)
                
                if (i+1)%50000 == 0 and i > 1:
                    cnt += 1
                    if not evecs:
                        jnp.save(f'{basepath2}/h_mat_'+str(seed)+'_'+str(cnt%2)+'.npy',h_mat)
                        jnp.save(f'{basepath2}/s_mat_'+str(seed)+'_'+str(cnt%2)+'.npy',s_mat)
                        '''if os.path.isfile(f'{basepath2}/h_mat_'+str(seed)+'_'+str(cnt%2)+'.npy'):
                            os.remove(f'{basepath2}/h_mat_'+str(seed)+'_'+str(cnt%2)+'.npy')
                        if os.path.isfile(f'{basepath2}/s_mat_'+str(seed)+'_'+str(cnt%2)+'.npy'):
                            os.remove(f'{basepath2}/s_mat_'+str(seed)+'_'+str(cnt%2)+'.npy')
                        shutil.copy(localpath + '/h_mat_'+str(save_seed)+'.npy',f'{basepath2}/h_mat_'+str(seed)+'_'+str(cnt%2)+'.npy')
                        shutil.copy(localpath + '/s_mat_'+str(save_seed)+'.npy',f'{basepath2}/s_mat_'+str(seed)+'_'+str(cnt%2)+'.npy')'''
                    else:
                        jnp.save(basepath2 + '/obs_'+str(save_seed)+'_'+str(cnt%2)+'.npy',obs_full)

                        #if os.path.isfile(f'{basepath2}/obs_'+str(seed)+'_'+str(cnt%2)+'.npy'):
                        #    os.remove(f'{basepath2}/obs_'+str(seed)+'_'+str(cnt%2)+'.npy')
                        #shutil.copy(localpath + '/obs_'+str(save_seed)+'.npy',f'{basepath2}/obs_'+str(seed)+'_'+str(cnt%2)+'.npy')
            #h_mat=s_mat=0.         

            
        if not save:
            #h_mat, s_mat = assemble_matrices(h_half, s_half, log_weight)
            if i > 0:
                if evecs:
                    obs = h5py.File(f"{basepath2}/obs_"+str(seed)+".h5",'r+')
                    dh = np.array(h_mat_f['h'])
                    h_mat_f['h'][...] = np.array(dh) + h_mat
                else:
                    h_mat_f = h5py.File(f"{basepath2}/h_mat.h5",'r+')
                    dh = np.array(h_mat_f['h'])
                    h_mat_f['h'][...] = np.array(dh) + h_mat
                    h_mat_f.close()

                    s_mat_f = h5py.File(f"{basepath2}/s_mat.h5",'r+')
                    ds = np.array(s_mat_f['s'])
                    s_mat_f['s'][...] = np.array(ds) + s_mat
                    s_mat_f.close()
                
            h_mat_f = h5py.File(f"{basepath2}/h_mat.h5",'w')
            h_mat_f.create_dataset('h',data=h_mat,chunks=True)
            h_mat_f.close()
            s_mat_f = h5py.File(f"{basepath2}/s_mat.h5",'w')
            s_mat_f.create_dataset('s',data=s_mat,chunks=True)
            s_mat_f.close()
            print('files saved!')
            np.save(f"{basepath2}/total_steps.npy", i)
            if evecs:
                jnp.save(f"{basepath2}/s_half.npy", s_half)
            jnp.save(f"{basepath2}/h_half.npy", h_half)
            jnp.save(f"{basepath2}/s_half.npy", s_half)
            print('matrices saved!')

        acc_rate = jnp.mean(mc_info["is_accepted"]).item()
        hacc_rate = 1. / jnp.mean(mc_info["recip_ratio"]).item()
        printer.print_fields({"step": i, "acc": acc_rate, "hacc": hacc_rate})
    if save:
        if not evecs:
            jnp.save(f'{basepath2}/h_mat_'+str(seed)+'_final.npy',h_mat)
            jnp.save(f'{basepath2}/s_mat_'+str(seed)+'_final.npy',s_mat)
            #shutil.copy(localpath + '/h_mat_'+str(save_seed)+'.npy',f'{basepath2}/h_mat_'+str(seed)+'_final.npy')
            #shutil.copy(localpath + '/s_mat_'+str(save_seed)+'.npy',f'{basepath2}/s_mat_'+str(seed)+'_final.npy')
            #if os.path.isfile(localpath + '/h_mat_'+str(seed)+'.npy'):
            #    os.remove(localpath + '/h_mat_'+str(seed)+'.npy')
            #if os.path.isfile(localpath + '/s_mat_'+str(seed)+'.npy'):
            #    os.remove(localpath + '/s_mat_'+str(seed)+'.npy')
        else:
            jnp.save(basepath2 + '/obs_final.npy',obs_full)
            #shutil.copy(localpath + '/obs_'+str(save_seed)+'.npy',f'{basepath2}/obs_'+str(seed)+'_final.npy')



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
    os.makedirs(save_folder+'/steps', exist_ok=False)
    # save config as yaml
    save_cfg_to_yaml(cfg, f"{save_folder}/hparams.yaml")

    key = jax.random.PRNGKey(cfg.seed) if 'seed' in cfg else None
    system, ansatz, frozen_ansatz, sampler, eval_fn, eval_fn_frozen, state, \
        = prepare(cfg.system, cfg.ansatz, cfg.sample, eval_cfg, cfg.restart, key)
    print('eval_obs:',cfg.eval_tangents.eval_obs)
    if cfg.eval_tangents.eval_obs:
        obs_args = {'cell':cfg.system.cell, 'spins':cfg.system.spin, 'bins':(48,)*2}
        evecs = cfg.eval_tangents.evecs
        if not evecs:
            raise NotImplementedError
    else:
        obs_args = None
        evecs = None

    run(sampler, eval_fn, eval_fn_frozen, state,
        eval_cfg.iterations, save_folder, [cfg.seed,cfg.save_seed], 
        eval_cfg.get("dense_mode", False), obs_args, evecs, eval_cfg.get("dev_2", False))

