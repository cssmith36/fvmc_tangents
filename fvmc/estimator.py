from functools import partial

import jax
import kfac_jax
from jax import lax
from jax import numpy as jnp

from .ewaldsum import EwaldSum
from .hamiltonian import calc_ke_elec, calc_ke_full, get_nuclei_mass, calc_pe
from .moire import OneShell
from .utils import ElecConf, FullConf, PMAP_AXIS_NAME, PmapAxis, exp_shifted


def get_log_psi(model_apply, params, stop_gradient=False):
    # make log of wavefunction from model.apply
    if stop_gradient:
        params = lax.stop_gradient(params)
    def log_psi(*args):
        sign, logd = model_apply(params, *args)
        if jnp.iscomplexobj(sign):
            logd += jnp.log(sign)
        return logd
    return log_psi


def parse_extpots(extpots, **sysinfo):
    """parse the extpots dict into a dict of functions"""
    extpots = extpots or {}
    parsed = {}
    for name, ext in extpots.items():
        if callable(ext):
            parsed[name] = ext
        elif name == 'moire':
            parsed[name] = OneShell(sysinfo['cell'], **ext).calc_pe
        else:
            raise ValueError(f'Unknown external potential: {name}')
    return parsed


def build_eval_local_elec(model, elems, nuclei, cell=None, *,
                          ke_kwargs=None, pe_kwargs=None,
                          extpots=None, stop_gradient=True):
    """create a function that evaluates local energy, sign and log abs of wavefunction.

    Args:
        model (nn.Module): a flax module that calculates the sign and log abs of wavefunction.
            `model.apply` should have signature (params, x) -> (sign(f(x)), log|f(x)|)
        elems (Array): the element indices (charges) of those nuclei.
        nuclei (Array): the position of nuclei.
        cell (Optional Array): if not None, using ewald summation for potential energy in PBC

    Returns:
        Callable with signature (params, x) -> (eloc, sign, logf) that evaluates
        local energy, sign and log abs of wavefunctions on given parameters and configurations.
    """

    ke_kwargs = ke_kwargs or {}
    ke_fn = partial(calc_ke_elec, **ke_kwargs)
    pe_kwargs = pe_kwargs or {}
    pe_fn = (EwaldSum(cell, **pe_kwargs).calc_pe
             if cell is not None else partial(calc_pe, **pe_kwargs))
    extpot_fns = parse_extpots(extpots, elems=elems, nuclei=nuclei, cell=cell)

    def eval_local(params, x: ElecConf):
        sign, logf = model.apply(params, x)
        log_psi_fn = get_log_psi(model.apply, params,
                                 stop_gradient=stop_gradient)
        ene_comps = {
            "e_kin": ke_fn(log_psi_fn, x),
            "e_coul": pe_fn(elems, nuclei, x),
            **{f"e_{name}": fn(x) for name, fn in extpot_fns.items()}
        }
        eloc = jax.tree_util.tree_reduce(jnp.add, ene_comps, 0.0)
        if stop_gradient:
            ene_comps = jax.tree_map(lax.stop_gradient, ene_comps)
            eloc = lax.stop_gradient(eloc)
        extras = {**ene_comps} # for now only log energy components
        return eloc, sign, logf, extras

    return eval_local


def build_eval_local_full(model, elems, cell=None, *,
                          ke_kwargs=None, pe_kwargs=None,
                          extpots=None, stop_gradient=True):
    """create a function that evaluates local energy, sign and log abs of full wavefunction.

    Args:
        model (nn.Module): a flax module that calculates the sign and log abs of wavefunction.
            `model.apply` should have signature (params, r, x) -> (sign(f), log|f|)
        elems (Array): the element indices (charges) of nuclei (corresponding to `r`).
        cell (Optional Array): if not None, using ewald summation for potential energy in PBC

    Returns:
        Callable with signature (params, conf) -> (eloc, sign, logf) that evaluates
        local energy, sign and log abs of wavefunctions on given parameters and conf.
        `conf` here is a tuple of (r, x) contains nuclei (r) and electron (x) positions.
    """

    mass = get_nuclei_mass(elems)
    ke_kwargs = ke_kwargs or {}
    ke_fn = partial(calc_ke_full, **ke_kwargs)
    pe_kwargs = pe_kwargs or {}
    pe_fn = (EwaldSum(cell, **pe_kwargs).calc_pe
             if cell is not None else partial(calc_pe, **pe_kwargs))
    extpot_fns = parse_extpots(extpots, elems=elems, cell=cell)

    def eval_local(params, conf: FullConf):
        r, x = conf
        sign, logf = model.apply(params, r, x)
        log_psi_fn = get_log_psi(model.apply, params,
                                 stop_gradient=stop_gradient)
        ene_comps = {
            "e_kin": ke_fn(log_psi_fn, mass, r, x),
            "e_coul": pe_fn(elems, r, x),
            **{f"e_{name}": fn(r, x) for name, fn in extpot_fns.items()}
        }
        eloc = jax.tree_util.tree_reduce(jnp.add, ene_comps, 0.0)
        if stop_gradient:
            ene_comps = jax.tree_map(lax.stop_gradient, ene_comps)
            eloc = lax.stop_gradient(eloc)
        extras = {**ene_comps} # for now only log energy components
        return eloc, sign, logf, extras

    return eval_local


def get_batched_local(eval_local_fn, mini_batch=None,
                      checkpoint=True, unroll_loop=False):
    if not mini_batch:
        return jax.vmap(eval_local_fn, in_axes=(None, 0), out_axes=0)
    def batch_local(params, data):
        partial_local = jax.vmap(partial(eval_local_fn, params), 0, 0)
        if checkpoint:
            partial_local = jax.checkpoint(partial_local, prevent_cse=False)
        stack_data = jax.tree_map(lambda x: x.reshape(
            x.shape[0] // mini_batch, mini_batch, *x.shape[1:]), data)
        if not unroll_loop:
            stack_res = lax.map(partial_local, stack_data)
            res = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), stack_res)
        else: # manually unroll the map to prevent a bug in compiling
            res_list = []
            n_iter, = set(jax.tree_util.tree_leaves(
                jax.tree_map(lambda a:a.shape[0], stack_data)))
            for ii in range(n_iter):
                slice_data = jax.tree_map(lambda a, ii=ii: a[ii], stack_data)
                res_list.append(partial_local(slice_data))
            res = jax.tree_map(lambda *a: jnp.concatenate(a, axis=0), *res_list)
        return res
    return batch_local


def clip_around(a, target, half_range, stop_gradient=True):
    if jnp.iscomplexobj(target):
        return (clip_around(a.real, target.real, half_range, stop_gradient)
                + 1j * clip_around(a.imag, target.imag, half_range, stop_gradient))
    c_min = target - half_range
    c_max = target + half_range
    if stop_gradient:
        c_max, c_min = map(lax.stop_gradient, (c_max, c_min))
    return jnp.clip(a, c_min, c_max)


def build_eval_total(eval_local_fn, energy_clipping=None,
                     clip_from_median=False, center_shifting=False,
                     mini_batch=None, checkpoint=True, unroll_loop=False,
                     pmap_axis_name=PMAP_AXIS_NAME,
                     use_weighted=False):
    """Create a function that evaluates quantities on the whole batch of samples.

    The created function will take paramters and sampled data as input,
    and returns a tuple with first element a loss that gives correct gradient to parameters,
    and the second element a dict contains multiple statistical quantities of the samples.

    Args:
        eval_local_fn (Callable): callable which evaluates
            the local energy, sign and log of absolute value of wfn.
        energy_clipping (float, optional): If greater than zero, clip local energies that are
            outside [E_t - n D, E_t + n D], where E_t is the mean local energy, n is
            this value and D the mean absolute deviation of the local energies.
            Defaults to None (no clipping).
        clip_from_median (bool): If true, center the clipping window at the median rather
            than the mean. Potentially expensive in multi-host training, but more
            accurate/robust to outliers. Defaults to False.
        center_shifting (bool): If True, shift the average local energy so that
            the mean difference of the batch is always zero. Will only be useful with
            effective local energy clipping. Defaults to True.
        pmap_axis_name (str): axis name used in pmap
        use_weighted (bool): If True, use `build_eval_total_weighted`, which will
            take the log of sample weight (`data[1]`) into account

    Returns:
        Callable with signature (params, data) -> (loss, aux) where data is a tuple of
        samples and corresponding probbility densities of the samples. loss is used in training
        the parameters that gives the correct gradient but its value is meaningless.
        aux is a dict that contains multiple statistical quantities calculated from the sample.
    """
    if use_weighted:
        return build_eval_total_weighted(
            eval_local_fn=eval_local_fn,
            energy_clipping=energy_clipping,
            clip_from_median=clip_from_median,
            center_shifting=center_shifting,
            mini_batch=mini_batch,
            checkpoint=checkpoint,
            unroll_loop=unroll_loop,
            pmap_axis_name=pmap_axis_name)

    paxis = PmapAxis(pmap_axis_name)
    batch_local = get_batched_local(eval_local_fn, mini_batch, checkpoint, unroll_loop)

    def eval_total(params, data):
        r"""return loss and statistical quantities calculated from samples.

        The loss is given in the following form:

            (\psi #[\psi^c (E_l^c - E_tot)]) / (#[\psi \psi^c]) + h.c.

        where \psi is the wavefunction, E_l is the (clipped) local energy,
        E_tot is the estimated total energy, ^c stands for conjugation,
        and #[...] stands for stop gradient. One can easily check that
        this form will give the correct gradient estimation.
        Note that instead of doing the h.c., we are using 2 * Real[...].
        """
        # data is a tuple of sample and log of sampling weight
        conf, _ = data
        eloc, sign, logf, extras = batch_local(params, conf)
        if jnp.iscomplexobj(sign):
            logf += jnp.log(sign)

        # compute total energy
        etot = paxis.all_nanmean(eloc)
        # compute variance
        var_e = paxis.all_nanmean(jnp.abs(eloc - etot)**2)
        # form aux data dict, divide tot_w for correct gradient
        aux = dict(e_tot = etot.real, var_e = var_e, std_e = jnp.sqrt(var_e),
                   nans = jnp.isnan(eloc).sum())
        for key, val in extras.items():
            aux[key] = paxis.all_nanmean(val).real

        # clipping the local energy (for making the loss)
        eclip = eloc
        if energy_clipping and energy_clipping > 0:
            ecenter = (jnp.median(paxis.all_gather(eloc.real))
                       if clip_from_median else etot.real)
            tv = paxis.all_nanmean(jnp.abs(eloc - etot))
            eclip = clip_around(eloc, ecenter, energy_clipping * tv, stop_gradient=True)
        # shift the constant and get diff
        ebar = paxis.all_nanmean(eclip) if center_shifting else etot
        ediff = lax.stop_gradient(eclip - ebar).conj()
        # 2 * real for h.c.
        kfac_jax.register_squared_error_loss(logf.real[:, None])
        loss = paxis.all_nanmean(2 * (logf * ediff).real)

        return loss, aux

    return eval_total


def build_eval_total_weighted(eval_local_fn, energy_clipping=None,
                              clip_from_median=False, center_shifting=True,
                              mini_batch=None, checkpoint=True, unroll_loop=False,
                              pmap_axis_name=PMAP_AXIS_NAME):
    """Create a function that evaluates quantities on the whole batch of samples.

    The created function will take paramters and sampled data as input,
    and returns a tuple with first element a loss that gives correct gradient to parameters,
    and the second element a dict contains multiple statistical quantities of the samples.

    Args:
        eval_local_fn (Callable): callable which evaluates
            the local energy, sign and log of absolute value of wfn.
        energy_clipping (float, optional): If greater than zero, clip local energies that are
            outside [E_t - n D, E_t + n D], where E_t is the mean local energy, n is
            this value and D the mean absolute deviation of the local energies.
            Defaults to None (no clipping).
        clip_from_median (bool): If true, center the clipping window at the median rather
            than the mean. Potentially expensive in multi-host training, but more
            accurate/robust to outliers. Defaults to False.
        center_shifting (bool): If True, shift the average local energy so that
            the mean difference of the batch is always zero. Will only be useful with
            effective local energy clipping. Defaults to True.
        pmap_axis_name (str): axis name used in pmap

    Returns:
        Callable with signature (params, data) -> (loss, aux) where data is a tuple of
        samples and corresponding probbility densities of the samples. loss is used in training
        the parameters that gives the correct gradient but its value is meaningless.
        aux is a dict that contains multiple statistical quantities calculated from the sample.
    """

    paxis = PmapAxis(pmap_axis_name)
    batch_local = get_batched_local(eval_local_fn, mini_batch, checkpoint, unroll_loop)

    def eval_total(params, data):
        r"""return loss and statistical quantities calculated from samples.

        The loss is given in the following form:

            (\psi #[\psi^c (E_l^c - E_tot)]) / (#[\psi \psi^c]) + h.c.

        where \psi is the wavefunction, E_l is the (clipped) local energy,
        E_tot is the estimated total energy, ^c stands for conjugation,
        and #[...] stands for stop gradient. One can easily check that
        this form will give the correct gradient estimation.
        Note that instead of doing the h.c., we are using 2 * Real[...].
        """
        # data is a tuple of sample and log of sampling weight
        conf, logsw = data
        logsw = lax.stop_gradient(logsw)
        eloc, sign, logf, extras = batch_local(params, conf)
        if jnp.iscomplexobj(sign):
            logf += jnp.log(sign)

        # calculating relative weights for stats
        rel_w, lshift = exp_shifted(2*logf.real - logsw,
            normalize="mean", pmap_axis_name=paxis.name)
        # compute total energy
        etot = paxis.all_nanaverage(eloc, rel_w)
        # compute variance
        var_e = paxis.all_nanaverage(jnp.abs(eloc - etot)**2, rel_w)
        # form aux data dict, divide tot_w for correct gradient
        aux = dict(
            e_tot = etot.real, var_e = var_e, std_e = jnp.sqrt(var_e),
            nans = jnp.isnan(eloc).sum(), _log_shift = lshift
        )
        for key, val in extras.items():
            aux[key] = paxis.all_nanmean(val).real

        # clipping the local energy (for making the loss)
        eclip = eloc
        if energy_clipping and energy_clipping > 0:
            ecenter = (jnp.median(paxis.all_gather(eloc.real))
                       if clip_from_median else etot)
            tv = paxis.all_nanaverage(jnp.abs(eloc - etot), rel_w)
            eclip = clip_around(eloc, ecenter, energy_clipping * tv, stop_gradient=True)
        # make the conjugated term (with stopped gradient)
        eclip_c, sign_c, logf_c = map(
            lambda x: lax.stop_gradient(x.conj()),
            (eclip, sign, logf))
        # make normalized psi_sqr, grad w.r.t it is equivalent to grad of log psi
        log_psi2_rel = logf + logf_c - logsw
        rel_w_d = lax.stop_gradient(rel_w) # detached
        psi_sqr = jnp.exp(log_psi2_rel - lshift)
        # shift the constant and get diff
        ebar = paxis.all_nanaverage(eclip, rel_w) if center_shifting else etot
        ediff = lax.stop_gradient(eclip_c - ebar.conj())
        # 2 * real for h.c.
        kfac_jax.register_squared_error_loss(psi_sqr.real[:, None])
        loss = (paxis.all_nansum(2 * (psi_sqr * ediff).real)
                / paxis.all_nansum(~jnp.isnan(ediff) * rel_w_d))

        return loss, aux

    return eval_total
