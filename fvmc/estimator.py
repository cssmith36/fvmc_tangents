from functools import partial

import jax
import kfac_jax
from jax import lax
from jax import numpy as jnp

from .hamiltonian import calc_local_energy
from .utils import PMAP_AXIS_NAME, PmapAxis, ith_output


def exp_shifted(x, normalize=None, pmap_axis_name=PMAP_AXIS_NAME):
    paxis = PmapAxis(pmap_axis_name)
    stblz = paxis.all_max(lax.stop_gradient(x))
    exp = jnp.exp(x - stblz)
    if normalize:
        assert normalize.lower() in ("sum", "mean"), "invalid normalize option"
        reducer = getattr(paxis, f"all_{normalize.lower()}")
        total = reducer(lax.stop_gradient(exp))
        exp /= total
        stblz += jnp.log(total)
    return exp, stblz


def clip_around(a, target, half_range, stop_gradient=True):
    c_min = target - half_range
    c_max = target + half_range
    if stop_gradient:
        c_max, c_min = map(lax.stop_gradient, (c_max, c_min))
    return jnp.clip(a, c_min, c_max)


def build_eval_local(model, ions, elems):
    """create a function that evaluates local energy, sign and log abs of wavefunction.
    
    The created function will calculate these quantities for both ket and bra (conjugated),
    resulting the returned array having an extra dimension of size 2 at end, for ket and bra.

    Args:
        model (nn.Module): a flax module that calculates the sign and log abs of wavefunction.
            `model.apply` should have signature (params, x) -> (sign(f(x)), log|f(x)|)
        ions (Array): the position of ions.
        elems (Array): the element indices (charges) of those ions.

    Returns:
        Callable with signature (params, x) -> (eloc, sign, logf) that evaluates 
        local energy, sign and log abs of wavefunctions on given parameters and configurations.
    """

    def eval_local(params, x):
        log_psi_abs = ith_output(partial(model.apply, params), 1)
        eloc = calc_local_energy(log_psi_abs, ions, elems, x)
        sign, logf = model.apply(params, x)
        return eloc, sign, logf

    return eval_local


def build_eval_total(eval_local_fn, energy_clipping=0., 
                     grad_stablizing=False, pmap_axis_name=PMAP_AXIS_NAME):
    """Create a function that evaluates quantities on the whole batch of samples.

    The created function will take paramters and sampled data as input,
    and returns a tuple with first element a loss that gives correct gradient to parameters,
    and the second element a dict contains multiple statistical quantities of the samples.

    Args:
        eval_local_fn (Callable): callable which evaluates 
            the local energy, sign and log of absolute value of wfn.
            Should return a tuple with shape ([..., 2], [..., 2], [..., 2]),
            where the last dim of size 2 corresponds to ket and bra (conj'd) results.
        energy_clipping (float): If greater than zero, clip local energies that are
            outside [E_t - n D, E_t + n D], where E_t is the mean local energy, n is
            this value and D the mean absolute deviation of the local energies.
            Defaults to 0 (no clipping).
        grad_stablizing (bool): If True, use a trick that substract the mean in the grad
            of log psi. This should give no contribution when there is no energy clipping
            because it is a constant times averaged E_loc - E_tot, which is zero. 
            But it will be helpful at the begining of training

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
        x, logsw = data
        logsw = lax.stop_gradient(logsw)
        eloc, sign, logf = batch_local(params, x)

        # calculating relative weights for stats
        # eloc_r, sign_r, logf_r = eloc.mean(-1), sign.prod(-1), logf.sum(-1)
        rel_w, lshift = exp_shifted(2*logf - logsw, 
            normalize="mean", pmap_axis_name=paxis.name)
        tot_w = paxis.all_mean(rel_w) # should be just 1, but provide correct gradient
        # compute total energy
        etot = paxis.all_mean((eloc) * rel_w).real / tot_w
        # compute variance
        var_e = paxis.all_mean(jnp.abs(eloc - etot)**2 * rel_w) / tot_w
        # form aux data dict, divide tot_w for correct gradient
        aux = dict(
            e_tot = etot, var_e = var_e, log_shift = lshift
        )

        # clipping the local energy (for making the loss)
        eclip = eloc
        if energy_clipping > 0:
            tv = paxis.all_mean(jnp.abs(eloc - etot).mean(-1) * rel_w) / tot_w
            eclip = clip_around(eloc, etot, energy_clipping * tv, stop_gradient=True)
        # make the conjugated term (with stopped gradient)
        eclip_c, sign_c, logf_c = map(
            lambda x: lax.stop_gradient(x.conj()), 
            (eclip, sign, logf))
        # make normalized psi_sqr, grad w.r.t it is equivalent to grad of log psi
        log_shifted = logf + logf_c - logsw
        if grad_stablizing: # substract the averaged log psi (like baseline)
            log_shifted -= paxis.all_mean(log_shifted)
        psi_sqr = jnp.exp(log_shifted)
        kfac_jax.register_squared_error_loss(psi_sqr[:, None])
        e_diff = lax.stop_gradient(eclip_c - etot)
        loss = paxis.all_mean(2 * (psi_sqr * e_diff).real) # 2 * real for h.c.
        
        return loss, aux

    return eval_total