from functools import partial

import jax
import kfac_jax
from jax import lax
from jax import numpy as jnp

from .hamiltonian import calc_local_energy
from .utils import PMAP_AXIS_NAME, PmapAxis, ith_output, exp_shifted


def clip_around(a, target, half_range, stop_gradient=True):
    c_min = target - half_range
    c_max = target + half_range
    if stop_gradient:
        c_max, c_min = map(lax.stop_gradient, (c_max, c_min))
    return jnp.clip(a, c_min, c_max)


def build_eval_local_elec(model, nuclei, elems):
    """create a function that evaluates local energy, sign and log abs of wavefunction.
    
    Args:
        model (nn.Module): a flax module that calculates the sign and log abs of wavefunction.
            `model.apply` should have signature (params, x) -> (sign(f(x)), log|f(x)|)
        nuclei (Array): the position of nuclei.
        elems (Array): the element indices (charges) of those nuclei.

    Returns:
        Callable with signature (params, x) -> (eloc, sign, logf) that evaluates 
        local energy, sign and log abs of wavefunctions on given parameters and configurations.
    """

    def eval_local(params, x):
        log_psi_abs = ith_output(partial(model.apply, params), 1)
        eloc = calc_local_energy(log_psi_abs, elems, nuclei, x)
        sign, logf = model.apply(params, x)
        return eloc, sign, logf

    return eval_local


def build_eval_local_full(model, elems):
    """create a function that evaluates local energy, sign and log abs of full wavefunction.
    
    Args:
        model (nn.Module): a flax module that calculates the sign and log abs of wavefunction.
            `model.apply` should have signature (params, r, x) -> (sign(f), log|f|)
        elems (Array): the element indices (charges) of nuclei (corresponding to `r`).

    Returns:
        Callable with signature (params, conf) -> (eloc, sign, logf) that evaluates 
        local energy, sign and log abs of wavefunctions on given parameters and conf.
        `conf` here is a tuple of (r, x) contains nuclei (r) and electron (x) positions.
    """

    def eval_local(params, conf):
        r, x = conf
        log_psi_abs = ith_output(partial(model.apply, params), 1)
        eloc = calc_local_energy(log_psi_abs, elems, r, x, ion_ke=True)
        sign, logf = model.apply(params, r, x)
        return eloc, sign, logf

    return eval_local


def build_eval_total(eval_local_fn, energy_clipping=None, 
                     grad_stablizing=False, pmap_axis_name=PMAP_AXIS_NAME, 
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
        grad_stablizing (bool): If True, use a trick that substract the mean in the grad
            of log psi. This should give no contribution when there is no energy clipping
            because it is a constant times averaged E_loc - E_tot, which is zero. 
            But it will be helpful at the begining of training
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
            eval_local_fn, 
            energy_clipping, 
            grad_stablizing, 
            pmap_axis_name)

    paxis = PmapAxis(pmap_axis_name)
    batch_local = jax.vmap(eval_local_fn, in_axes=(None, 0), out_axes=0)

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
        eloc, sign, logf = batch_local(params, conf)

        # compute total energy
        etot = paxis.all_mean(eloc.real)
        # compute variance
        var_e = paxis.all_mean(jnp.abs(eloc - etot)**2)
        # form aux data dict, divide tot_w for correct gradient
        aux = dict(e_tot = etot, var_e = var_e)

        # clipping the local energy (for making the loss)
        eclip = eloc
        if energy_clipping and energy_clipping > 0:
            tv = paxis.all_mean(jnp.abs(eloc - etot).mean(-1))
            eclip = clip_around(eloc, etot, energy_clipping * tv, stop_gradient=True)
        # make the conjugated term (with stopped gradient)
        eclip_c = lax.stop_gradient(eclip.conj())
        # shift the constant
        if grad_stablizing:
            logf -= paxis.all_mean(logf)
        kfac_jax.register_squared_error_loss(logf[:, None])
        e_diff = lax.stop_gradient(eclip_c - etot)
        # 2 * real for h.c.
        loss = paxis.all_mean(2 * (logf * e_diff).real)
        
        return loss, aux

    return eval_total


def build_eval_total_weighted(eval_local_fn, energy_clipping=None, 
                              grad_stablizing=False, pmap_axis_name=PMAP_AXIS_NAME):
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
        grad_stablizing (bool): If True, use a trick that substract the mean in the grad
            of log psi. This should give no contribution when there is no energy clipping
            because it is a constant times averaged E_loc - E_tot, which is zero. 
            But it will be helpful at the begining of training
        pmap_axis_name (str): axis name used in pmap

    Returns:
        Callable with signature (params, data) -> (loss, aux) where data is a tuple of
        samples and corresponding probbility densities of the samples. loss is used in training
        the parameters that gives the correct gradient but its value is meaningless.
        aux is a dict that contains multiple statistical quantities calculated from the sample.
    """

    paxis = PmapAxis(pmap_axis_name)
    batch_local = jax.vmap(eval_local_fn, in_axes=(None, 0), out_axes=0)

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
        eloc, sign, logf = batch_local(params, conf)

        # calculating relative weights for stats
        rel_w, lshift = exp_shifted(2*logf.real - logsw, 
            normalize="mean", pmap_axis_name=paxis.name)
        # compute total energy
        etot = paxis.all_average(eloc.real, rel_w)
        # compute variance
        var_e = paxis.all_average(jnp.abs(eloc - etot)**2, rel_w)
        # form aux data dict, divide tot_w for correct gradient
        aux = dict(
            e_tot = etot, var_e = var_e, _log_shift = lshift
        )

        # clipping the local energy (for making the loss)
        eclip = eloc
        if energy_clipping and energy_clipping > 0:
            tv = paxis.all_average(jnp.abs(eloc - etot).mean(-1), rel_w)
            eclip = clip_around(eloc, etot, energy_clipping * tv, stop_gradient=True)
        # make the conjugated term (with stopped gradient)
        eclip_c, sign_c, logf_c = map(
            lambda x: lax.stop_gradient(x.conj()), 
            (eclip, sign, logf))
        # make normalized psi_sqr, grad w.r.t it is equivalent to grad of log psi
        log_psi2_rel = logf + logf_c - logsw
        rel_w_d = lax.stop_gradient(rel_w) # detached
        if grad_stablizing: # substract the averaged log psi (like baseline)
            log_psi2_rel -= paxis.all_average(log_psi2_rel, rel_w_d)
        psi_sqr = jnp.exp(log_psi2_rel - lshift)
        kfac_jax.register_squared_error_loss(psi_sqr[:, None])
        e_diff = lax.stop_gradient(eclip_c - etot)
        # 2 * real for h.c.
        loss = paxis.all_mean(2 * (psi_sqr * e_diff).real) / paxis.all_mean(rel_w_d)
        
        return loss, aux

    return eval_total