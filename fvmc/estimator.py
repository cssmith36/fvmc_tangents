from functools import partial

import jax
import kfac_jax
from jax import lax
from jax import numpy as jnp

from .ewaldsum import EwaldSum
from .hamiltonian import calc_ke_elec, calc_ke_full, get_nuclei_mass, calc_pe
from .utils import PMAP_AXIS_NAME, PmapAxis, ith_output, exp_shifted


def clip_around(a, target, half_range, stop_gradient=True):
    if jnp.iscomplexobj(a):
        return (clip_around(a.real, target.real, half_range, stop_gradient)
                + 1j * clip_around(a.imag, target.imag, half_range, stop_gradient))
    c_min = target - half_range
    c_max = target + half_range
    if stop_gradient:
        c_max, c_min = map(lax.stop_gradient, (c_max, c_min))
    return jnp.clip(a, c_min, c_max)


def build_eval_local_elec(model, elems, nuclei, cell=None):
    """create a function that evaluates local energy, sign and log abs of wavefunction.
    
    Args:
        model (nn.Module): a flax module that calculates the sign and log abs of wavefunction.
            `model.apply` should have signature (params, x) -> (sign(f(x)), log|f(x)|)
        nuclei (Array): the position of nuclei.
        elems (Array): the element indices (charges) of those nuclei.
        cell (Optional Array): if not None, using ewald summation for potential energy in PBC

    Returns:
        Callable with signature (params, x) -> (eloc, sign, logf) that evaluates 
        local energy, sign and log abs of wavefunctions on given parameters and configurations.
    """
    
    calc_pe_adapt = EwaldSum(cell).calc_pe if cell is not None else calc_pe

    def eval_local(params, x):
        log_psi_abs = ith_output(partial(model.apply, params), 1)
        ke = calc_ke_elec(log_psi_abs, x)
        pe = calc_pe_adapt(elems, nuclei, x)
        eloc = ke + pe
        sign, logf = model.apply(params, x)
        return eloc, sign, logf

    return eval_local


def build_eval_local_full(model, elems, cell=None):
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

    calc_pe_adapt = EwaldSum(cell).calc_pe if cell is not None else calc_pe
    mass = get_nuclei_mass(elems)

    def eval_local(params, conf):
        r, x = conf
        log_psi_abs = ith_output(partial(model.apply, params), 1)
        ke = calc_ke_full(log_psi_abs, mass, r, x)
        pe = calc_pe_adapt(elems, r, x)
        eloc = ke + pe
        sign, logf = model.apply(params, r, x)
        return eloc, sign, logf

    return eval_local


def build_eval_total(eval_local_fn, energy_clipping=None, 
                     center_shifting=False, pmap_axis_name=PMAP_AXIS_NAME, 
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
            center_shifting, 
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
        etot = paxis.all_nanmean(eloc)
        # compute variance
        var_e = paxis.all_nanmean(jnp.abs(eloc - etot)**2)
        # form aux data dict, divide tot_w for correct gradient
        aux = dict(e_tot = etot.real, var_e = var_e, std_e = jnp.sqrt(var_e),
                   nans = jnp.isnan(eloc).sum())

        # clipping the local energy (for making the loss)
        eclip = eloc
        if energy_clipping and energy_clipping > 0:
            tv = paxis.all_nanmean(jnp.abs(eloc - etot))
            eclip = clip_around(eloc, etot, energy_clipping * tv, stop_gradient=True)
        # shift the constant and get diff
        ebar = paxis.all_nanmean(eclip) if center_shifting else etot
        ediff = lax.stop_gradient(eclip - ebar).conj()
        # 2 * real for h.c.
        kfac_jax.register_squared_error_loss(logf[:, None])
        loss = paxis.all_nanmean(2 * (logf * ediff).real)
        
        return loss, aux

    return eval_total


def build_eval_total_weighted(eval_local_fn, energy_clipping=None, 
                              center_shifting=False, pmap_axis_name=PMAP_AXIS_NAME):
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
        etot = paxis.all_nanaverage(eloc, rel_w)
        # compute variance
        var_e = paxis.all_nanaverage(jnp.abs(eloc - etot)**2, rel_w)
        # form aux data dict, divide tot_w for correct gradient
        aux = dict(
            e_tot = etot.real, var_e = var_e, std_e = jnp.sqrt(var_e),
            nans = jnp.isnan(eloc).sum(), _log_shift = lshift
        )

        # clipping the local energy (for making the loss)
        eclip = eloc
        if energy_clipping and energy_clipping > 0:
            tv = paxis.all_nanaverage(jnp.abs(eloc - etot), rel_w)
            eclip = clip_around(eloc, etot, energy_clipping * tv, stop_gradient=True)
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
        kfac_jax.register_squared_error_loss(psi_sqr[:, None])
        loss = (paxis.all_nansum(2 * (psi_sqr * ediff).real) 
                / paxis.all_nansum(~jnp.isnan(ediff) * rel_w_d))
        
        return loss, aux

    return eval_total