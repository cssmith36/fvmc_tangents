from functools import partial
from typing import Callable, Optional, NamedTuple, Union, Tuple

import jax
from jax import lax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from optax import Updates, GradientTransformationExtraArgs

from .utils import Array, ArrayTree, PMAP_AXIS_NAME, PmapAxis, exp_shifted


Data = Union[ArrayTree, Tuple[ArrayTree, Array]]


class FisherPrecondState(NamedTuple):
    """State for fisher preconditioner, logging last step ncg"""
    last_grads_flat: Updates
    mixing_factor: float


def build_fisher_preconditioner(
        log_prob_fn: Callable,
        damping: float = 1e-3,
        maxiter: Optional[int] = None,
        mixing_factor: float = 0.,
        pmap_axis_name: str = PMAP_AXIS_NAME,
        use_weighted: bool = False,
) -> GradientTransformationExtraArgs:
    r"""build a preconditioner apply inverse fisher to the grad.
    
    This function will return a function that can be called as a preconditioner.
    Given grad, the function will return 

        (0.25 * F + damping * I)^{-1} grad,
    
    where F is the Fisher information matrix, calculated as the correlation of 
    the score of parameters: \partial log p / \partial theta_i
    """
    # this method use conjugate gradient to inverse the fisher matrix
    # only fisher vector product is lazily evaluated in the cg iteration
    # check https://gebob19.github.io/natural-gradient/
    # and https://github.com/jeffminlin/vmcnet/blob/master/vmcnet/updates/sr.py
    # and https://github.com/n-gao/pesnet/blob/main/pesnet/utils/optim.py

    paxis = PmapAxis(pmap_axis_name)

    def init_fn(params):
        return FisherPrecondState(
            last_grads_flat=jax.tree_map(jnp.zeros_like, ravel_pytree(params)[0]),
            mixing_factor=0.0)

    def update_fn(grads, state, params, data):
        """precondition function that apply inverse fisher matrix to gradients"""

        # handle (potential) sample weights
        if not (isinstance(data, Tuple) and len(data) == 2):
            data = (data, None)
        sample, logsw = data

        # flat log_p functions
        grads_flat, unravel_fn = ravel_pytree(grads)
        params_flat, _ = ravel_pytree(params)
        raveled_logpsi = lambda p_flat: 0.5 * log_prob_fn(unravel_fn(p_flat), sample)

        # logp.shape == (n_sample,)
        logpsi, vjp_fn = jax.vjp(raveled_logpsi, params_flat)
        logpsi_, jvp_fn = jax.linearize(raveled_logpsi, params_flat)

        # paxis.mean(jnp.sum(rel_w)) == 1
        rel_w = (exp_shifted(lax.stop_gradient(2 * logpsi - logsw), 
                             normalize="mean", pmap_axis_name=paxis.name)[0]
                 if use_weighted and logsw is not None else 1.)
        rel_w /= logpsi.shape[0] # so we will use sum for local batch (n_sample) dim
        
        def fisher_apply(x): # (damped) fisher vector product
            # x has the same shape as grad (raveled)
            jvp = jvp_fn(x) # shape = (n_sample,) same as logp
            mean_jvp = paxis.pmean(jnp.sum(jvp * rel_w, axis=0))
            jvp_centered = jvp - mean_jvp
            fvp_local, = vjp_fn(jvp_centered * rel_w)
            fvp = paxis.pmean(fvp_local) # local sum is done by vjp
            return fvp + damping * x
        
        mix = state.mixing_factor
        init_guess = state.last_grads_flat * mix + grads_flat * (1 - mix)
        precond_grads_flat, _ = jax.scipy.sparse.linalg.cg(
            fisher_apply,
            grads_flat,
            x0=init_guess,
            maxiter=maxiter)
        
        new_state = FisherPrecondState(precond_grads_flat, mixing_factor)
        return unravel_fn(precond_grads_flat), new_state

    return GradientTransformationExtraArgs(init_fn, update_fn)