from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import optax
from jax import lax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from optax import GradientTransformationExtraArgs, ScalarOrSchedule

from .utils import (PMAP_AXIS_NAME, Array, PmapAxis, adaptive_grad, chol_qr,
                    exp_shifted, fast_svd)


def fisher_inv_direct(score, vec, damping, state=None, *, paxis=None):
    # score: n_sample x n_params
    fisher = (score.T @ score.conj()).real
    if paxis is not None:
        fisher = paxis.pmean(fisher)
    fisher += damping * jnp.eye(fisher.shape[0])
    return jax.scipy.linalg.solve(fisher, vec), state


def fisher_inv_qr(score, vec, damping, state=None, *, paxis=None):
    # score: n_sample (n) x n_params (m)
    if jnp.iscomplexobj(score):
        score = jnp.concatenate([score.real, score.imag], axis=0)
    # gather all samples together
    score = paxis.all_gather(score, axis=0, tiled=True)
    # q: m x n, r: n x n
    q, r = chol_qr(score.T, shift=damping)
    # v ~= (S @ S.T + damping * jnp.eye(m))^-1 @ vec
    v = 1./damping * (vec - q @ (q.T @ vec))
    return v, state


def fisher_inv_svd(score, vec, damping, state=None, *, paxis=None):
    # score: n_sample (n) x n_params (m)
    if jnp.iscomplexobj(score):
        score = jnp.concatenate([score.real, score.imag], axis=0)
    # gather all samples together
    score = paxis.all_gather(score, axis=0, tiled=True)
    # vh: n x m
    _, s, vh = fast_svd(score)
    s2inv = 1. / (s**2 + damping)
    v = (jnp.einsum("ia,a,aj,j->i", vh.T, s2inv, vh, vec)
         + 1./damping * (vec - vh.T.dot(vh.dot(vec))))
    return v, state


def fisher_inv_iter(score, vec, damping, state, *, paxis=None,
                    solver='cg', precondition=False,
                    tol=1e-10, maxiter=100, x0_mixing=0.):
    # score: n_sample (n) x n_params (m)
    def fisher_apply(x):
        fvp = (score.T.conj() @ (score @ x)).real
        if paxis is not None:
            fvp = paxis.pmean(fvp)
        return fvp + damping * x
    # M is the preconditioning matrix
    M = None
    if precondition:
        fisher_diag = jnp.einsum("np,np->p", score.conj(), score).real
        M = lambda x: x / (fisher_diag + damping)
    return _stateful_iter_solve(fisher_apply, vec, state,
                solver=solver, tol=tol, maxiter=maxiter, x0_mixing=x0_mixing, M=M)


def constrain_norm(precond_grads, raw_grads, max_norm):
    gnorm2 = jnp.sum(precond_grads * raw_grads)
    gnorm = jnp.sqrt(jnp.clip(gnorm2, a_min=1e-12))
    return precond_grads * jnp.minimum(max_norm/gnorm, 1)


class FisherPrecondState(NamedTuple):
    """State for fisher preconditioner, logging step and optional solver state"""
    count: int
    solver_state: Any


def scale_by_fisher_inverse(
        log_psi_fn: Callable,
        mode: str = "qr",
        damping: ScalarOrSchedule = 1e-3,
        shifting: ScalarOrSchedule = 1.,
        max_norm: ScalarOrSchedule = 3e-2,
        use_weighted: bool = False,
        pmap_axis_name: str = PMAP_AXIS_NAME,
        **solver_kwargs
) -> GradientTransformationExtraArgs:
    r"""build a preconditioner apply inverse fisher to the grad.

    This function will return a function that can be called as a preconditioner.
    Given grad, the function will return

        (F + damping * I)^{-1} grad,

    where F is the Fisher information matrix, calculated as the correlation of
    the score of parameters: \partial log \psi / \partial theta_i.
    Note here \psi is the wavefunction and is actually sqrt of the probability.
    """
    # check https://gebob19.github.io/natural-gradient/
    # and https://github.com/jeffminlin/vmcnet/blob/master/vmcnet/updates/sr.py
    # and https://github.com/n-gao/pesnet/blob/main/pesnet/utils/optim.py

    paxis = PmapAxis(pmap_axis_name)

    mode = mode.lower()
    solver = mode.removeprefix("lazy_")
    eager_fisher_inv = (
        fisher_inv_direct if solver == "direct" else
        fisher_inv_qr if solver == "qr" else
        fisher_inv_svd if solver == "svd" else
        partial(fisher_inv_iter, solver=solver))

    _get_damping = _ensure_schedule(damping)
    _get_shift_factor = _ensure_schedule(shifting)
    _get_max_norm = _ensure_schedule(max_norm)

    def init_fn(params):
        solver_state = (_IterSolverState(
                x_old=jnp.zeros_like(ravel_pytree(params)[0]), mix=0.)
            if solver.lower() not in ("direct", "qr", "svd") else None)
        return FisherPrecondState(count=0, solver_state=solver_state)

    # eager means the score matrix will be directly calculated and stored
    # only one of the update fn will be used (eager or lazy)
    def eager_update_fn(grads, state, params, data):
        """precondition function that apply inverse fisher matrix to gradients"""

        # handle (potential) sample weights
        if not (isinstance(data, Tuple) and len(data) == 2):
            data = (data, None)
        sample, logsw = data
        n_sample = sample.shape[0]

        # flat log_p functions
        grads_flat, unravel_fn = ravel_pytree(grads)
        score_fn = jax.vmap(
            adaptive_grad(log_psi_fn, argnums=0),
            in_axes=[None, 0], out_axes=0)
        score = jax.vmap(lambda p: ravel_pytree(p)[0])(
            score_fn(params, sample))

        # paxis.mean(jnp.sum(rel_w)) == 1, and rel_w has to be positive
        rel_w = jnp.ones(n_sample)
        if use_weighted and logsw is not None:
            logpsi = jax.vmap(log_psi_fn, in_axes=[None, 0])(params, sample)
            rel_w = exp_shifted(lax.stop_gradient(2 * logpsi.real - logsw),
                                normalize="mean", pmap_axis_name=paxis.name)[0]
        rel_w /= n_sample # so we will use sum for local batch (n_sample) dim

        count, solver_state = state
        damping = _get_damping(count)
        shift_factor = _get_shift_factor(count)
        center_factor = 1. - jnp.sqrt(1. - shift_factor)
        max_norm = _get_max_norm(count)

        mean_score = paxis.pmean(jnp.sum(rel_w[:, None] * score, axis=0))
        score = score - center_factor * mean_score
        score = score * jnp.sqrt(rel_w[:, None])

        precond_grads_flat, new_solver_state = eager_fisher_inv(
            score, grads_flat, damping, solver_state,
            paxis=paxis, **solver_kwargs)
        new_state = FisherPrecondState(count+1, new_solver_state)
        cgrads_flat = constrain_norm(precond_grads_flat, grads_flat, max_norm)
        return unravel_fn(cgrads_flat), new_state

    # lazy means fisher vector product is lazily evaluated in the cg iteration
    # this is only useful when the score matrix is huge and cannot be stored
    def lazy_update_fn(grads, state, params, data):
        """precondition function that apply inverse fisher matrix to gradients"""

        # handle (potential) sample weights
        if not (isinstance(data, Tuple) and len(data) == 2):
            data = (data, None)
        sample, logsw = data

        # flat log_p functions
        grads_flat, unravel_fn = ravel_pytree(grads)
        params_flat, _ = ravel_pytree(params)
        batched_logp = jax.vmap(log_psi_fn, (None, 0))
        raveled_logp = lambda p_flat: batched_logp(unravel_fn(p_flat), sample)

        # logpsi.shape == (n_sample,)
        logpsi, vjp_fn = jax.vjp(raveled_logp, params_flat)
        logpsi_, jvp_fn = jax.linearize(raveled_logp, params_flat)
        assert logpsi.ndim == 1

        count, solver_state = state
        damping = _get_damping(count)
        shift_factor = _get_shift_factor(count)
        max_norm = _get_max_norm(count)

        # paxis.mean(jnp.sum(rel_w)) == 1
        rel_w = (exp_shifted(lax.stop_gradient(2 * logpsi.real - logsw),
                             normalize="mean", pmap_axis_name=paxis.name)[0]
                 if use_weighted and logsw is not None else 1.)
        rel_w /= logpsi.shape[0] # so we will use sum for local batch (n_sample) dim

        def fisher_apply(x): # (damped) fisher vector product
            # x has the same shape as grad (raveled)
            jvp = jvp_fn(x.conj()) # shape = (n_sample,) same as logp
            mean_jvp = paxis.pmean(jnp.sum(jvp * rel_w, axis=0))
            jvp = jvp - shift_factor * mean_jvp
            fvp_local, = vjp_fn(jvp.conj() * rel_w)
            fvp = paxis.pmean(fvp_local.real) # local sum is done by vjp
            return fvp + damping * x

        precond_grads_flat, new_solver_state = _stateful_iter_solve(
            fisher_apply,
            grads_flat,
            solver_state,
            solver=solver,
            M=None,
            **solver_kwargs)
        new_state = FisherPrecondState(count+1, new_solver_state)
        cgrads_flat = constrain_norm(precond_grads_flat, grads_flat, max_norm)
        return unravel_fn(cgrads_flat), new_state

    # choose only one update function
    update_fn = lazy_update_fn if mode.startswith("lazy_") else eager_update_fn

    return GradientTransformationExtraArgs(init_fn, update_fn)


class _IterSolverState(NamedTuple):
    x_old: Array
    mix: float

def _stateful_iter_solve(A, b, state, *, solver="cg", x0_mixing=0., **kwargs):
    solver = getattr(jax.scipy.sparse.linalg, solver)
    x_old, mix = state
    x0 = x_old * mix + b * (1 - mix)
    x, _ = solver(A, b, x0, **kwargs)
    new_state = _IterSolverState(x, x0_mixing)
    return x, new_state


def _ensure_schedule(number_or_schedule):
    if not callable(number_or_schedule):
        return lambda i: number_or_schedule
    return number_or_schedule
