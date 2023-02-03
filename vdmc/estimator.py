import jax
from jax import lax
from jax import numpy as jnp
from functools import partial
import kfac_jax

from .utils import paxis, ith_output
from .hamiltonian import calc_kinetic_energy, calc_potential_energy


def exp_shifted(x, normalize=None):
    stblz = paxis.all_max(x)
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
        log_psi_fn = ith_output(partial(model.apply, params), 1)
        eloc = (calc_kinetic_energy(log_psi_fn, x) 
                + calc_potential_energy(ions, elems, x))
        sign, logf = model.apply(params, x)
        return (jnp.stack([eloc, eloc.conj()], -1),
                jnp.stack([sign, sign.conj()], -1),
                jnp.stack([logf, logf.conj()], -1))

    return eval_local


def build_eval_total(eval_local_fn, clipping=0.):
    """Create a function that evaluates quantities on the whole batch of samples.

    The created function will take paramters and sampled data as input,
    and returns a tuple with first element a loss that gives correct gradient to parameters,
    and the second element a dict contains multiple statistical quantities of the samples.

    Args:
        eval_local_fn (Callable): callable which evaluates the local energy, sign and log abs of wfn.
            Should return a tuple with shape ([..., 2], [..., 2], [..., 2]),
            where the last dim of size 2 corresponds to ket and bra (conj'd) results.
        clipping (float): If greater than zero, clip local energies that are
            outside [E_t - n D, E_t + n D], where E_t is the mean local energy, n is
            this value and D the mean absolute deviation of the local energies.

    Returns:
        Callable with signature (params, data) -> (loss, aux) where data is a tuple of
        samples and corresponding probbility densities of the samples. loss is used in training
        the parameters that gives the correct gradient but its value is meaningless.
        aux is a dict that contains multiple statistical quantities calculated from the sample.
    """

    batch_local = jax.vmap(eval_local_fn, in_axes=(None, 0), out_axes=0)

    def eval_total(params, data):
        r"""return loss and statistical quantities calculated from samples.

        The loss is given in the following form:

            (\psi #[\psi^c (E_l^c - E_tot)]) / (#[\psi \psi^c]) + h.c.

        where \psi is the wavefunction, E_l is the (clipped) local energy, 
        E_tot is the estimated total energy, ^c stands for conjugation,
        and #[...] stands for stop gradient. One can easily check that 
        this form will give the correct gradient estimation.        
        """
        # data is a tuple of sample and log of sampling weight
        x, logsw = data
        logsw = lax.stop_gradient(logsw)
        eloc, sign, logf = batch_local(params, x)

        # calculating relative weights for stats
        eloc_r, sign_r, logf_r = eloc.mean(-1), sign.prod(-1), logf.sum(-1)
        rel_w, lshift = exp_shifted(logf_r - logsw, normalize="mean")
        tot_w = paxis.all_mean(rel_w) # should be just 1, but provide correct gradient
        # compute averages and total energy
        avg_es = paxis.all_mean((eloc_r * sign_r) * rel_w).real # averaged (sign * eloc)
        avg_s = paxis.all_mean(sign_r * rel_w).real # averaged sign
        etot = avg_es / avg_s
        # compute variances
        var_e = paxis.all_mean(jnp.abs(eloc_r - etot)**2 * sign_r * rel_w) / avg_s
        var_s = paxis.all_mean(jnp.abs(sign_r - avg_s/tot_w)**2 * rel_w) / tot_w
        # form aux data dict, divide tot_w for correct gradient
        aux = dict(
            e_tot = etot, avg_es = avg_es/tot_w, avg_s = avg_s/tot_w,
            var_e = var_e, var_s = var_s, log_shift = lshift
        )

        # clipping the local energy (for making the loss)
        eclip = eloc
        if clipping > 0:
            tv = paxis.all_mean(jnp.abs(eloc - etot).mean(-1) * rel_w) / tot_w
            eclip = clip_around(eloc, etot, clipping * tv, stop_gradient=True)
        # make the conjugated term (with stopped gradient)
        eclip_c, sign_c, logf_c = map(
            lambda x: lax.stop_gradient(jnp.flip(x, -1)), 
            (eclip, sign, logf))
        # make normalized psi_sqr, grad w.r.t it is equivalent to grad of log psi
        psi_sqr = (jnp.exp(logf + logf_c - logsw[..., None])
                   * sign * sign_c / lax.stop_gradient(avg_s))
        kfac_jax.register_squared_error_loss(psi_sqr, weight=0.5)
        e_diff = lax.stop_gradient(eclip_c - etot)
        loss = paxis.all_mean((psi_sqr * e_diff).sum(-1)) # sum(-1) for h.c.
        
        return loss, aux

    return eval_total