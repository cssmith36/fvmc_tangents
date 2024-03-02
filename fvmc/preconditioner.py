from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import optax
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax.flatten_util import ravel_pytree
from optax import GradientTransformationExtraArgs, ScalarOrSchedule

from .utils import (PMAP_AXIS_NAME, Array, PmapAxis, adaptive_grad, chol_qr,
                    exp_shifted, fast_svd)


def constrain_norm(precond_grads, raw_grads, max_norm):
    gnorm2 = jnp.sum(precond_grads * raw_grads)
    gnorm = jnp.sqrt(jnp.clip(gnorm2, a_min=1e-12))
    return precond_grads * jnp.minimum(max_norm/gnorm, 1)


class FisherPrecondState(NamedTuple):
    """State for fisher preconditioner, logging step and optional solver state"""
    count: int
    solver_state: Any
    prox_grads: Optional[Array] = None


def scale_by_fisher_inverse(
        log_psi_fn: Callable,
        mode: str = "chol",
        damping: ScalarOrSchedule = 1e-3,
        shifting: ScalarOrSchedule = 1.,
        max_norm: ScalarOrSchedule = 3e-2,
        proximal: Optional[ScalarOrSchedule] = None,
        use_weighted: bool = False,
        pmap_axis_name: str = PMAP_AXIS_NAME,
        mini_batch: Optional[int] = None,
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
        fisher_inv_svd if solver == "svd" else
        fisher_inv_qr if solver == "qr" else
        fisher_inv_chol if solver == "chol" else
        partial(fisher_inv_iter, solver=solver))

    _get_damping = _ensure_schedule(damping)
    _get_shift_factor = _ensure_schedule(shifting)
    _get_max_norm = _ensure_schedule(max_norm)
    _get_proximal = _ensure_schedule(proximal)

    def init_fn(params):
        solver_state = (_IterSolverState(
                x_old=jnp.zeros_like(ravel_pytree(params)[0]), mix=0.)
            if solver.lower() not in ("direct", "qr", "svd") else None)
        prox_grads = (jnp.zeros_like(ravel_pytree(params)[0])
                      if proximal is not None else None)
        return FisherPrecondState(count=0, solver_state=solver_state,
                                  prox_grads=prox_grads)

    def update_fn(grads, state, params, data):
        """precondition function that apply inverse fisher matrix to gradients"""

        count, solver_state, prox_grads = state
        damping = _get_damping(count)
        shift_factor = _get_shift_factor(count)
        max_norm = _get_max_norm(count)
        proximal = _get_proximal(count)

        # flat log_p functions
        grads_flat, unravel_fn = ravel_pytree(grads)
        if proximal is not None:
            grads_flat += damping * proximal * prox_grads
        # handle (potential) sample weights
        if not (isinstance(data, Tuple) and len(data) == 2):
            data = (data, None)
        sample, logsw = data
        n_sample = jax.tree_leaves(sample)[0].shape[0]
        # make sure paxis.sum(jnp.sum(rel_w)) == 1, and rel_w has to be positive
        rel_w = jnp.ones((1,))
        if use_weighted and logsw is not None:
            logpsi = jax.vmap(log_psi_fn, in_axes=[None, 0])(params, sample)
            rel_w = exp_shifted(lax.stop_gradient(2 * logpsi.real - logsw),
                                normalize="mean", pmap_axis_name=paxis.name)[0]
        rel_w /= (n_sample * paxis.size()) # so we always use sum for batch dim

        inner_fn = _lazy_inner if mode.startswith("lazy") else _eager_inner
        precond_grads_flat, new_solver_state = inner_fn(
            grads_flat, solver_state, params, sample, rel_w,
            damping, shift_factor)

        new_prox_grads = precond_grads_flat if proximal is not None else None
        new_state = FisherPrecondState(count+1, new_solver_state, new_prox_grads)
        cgrads_flat = constrain_norm(precond_grads_flat, grads_flat, max_norm)
        return unravel_fn(cgrads_flat), new_state

    # eager means the score matrix will be directly calculated and stored
    # only one of the update fn will be used (eager or lazy)
    def _eager_inner(grads_flat, solver_state, params, sample, rel_w,
                     damping, shift_factor):
        _grad_fn = adaptive_grad(log_psi_fn, argnums=0)
        score_fn = jax.vmap(lambda s: ravel_pytree(_grad_fn(params, s))[0])
        if mini_batch is None:
            score = score_fn(sample)
        else:
            batch_sample = jax.tree_map(
                lambda s: s.reshape(-1, mini_batch, *s.shape[1:]), sample)
            batch_score = lax.map(score_fn, batch_sample)
            score = batch_score.reshape(-1, grads_flat.shape[-1])
        # paxis.sum(jnp.sum(rel_w)) == 1, and rel_w has to be positive
        mean_score = paxis.psum(jnp.sum(rel_w[:, None] * score, axis=0))
        center_factor = 1. - jnp.sqrt(1. - shift_factor)
        score = score - center_factor * mean_score
        score = score * jnp.sqrt(rel_w[:, None])
        # precond_grads_flat: n_params
        precond_grads_flat, new_solver_state = eager_fisher_inv(
            score, grads_flat, damping, solver_state,
            paxis=paxis, **solver_kwargs)
        return precond_grads_flat, new_solver_state

    # lazy means fisher vector product is lazily evaluated in the cg iteration
    # this is only useful when the score matrix is huge and cannot be stored
    def _lazy_inner(grads_flat, solver_state, params, sample, rel_w,
                    damping, shift_factor):
        params_flat, unravel_fn = ravel_pytree(params)
        batched_logp = jax.vmap(log_psi_fn, (None, 0))
        raveled_logp = lambda p_flat: batched_logp(unravel_fn(p_flat), sample)
        # logpsi.shape == (n_sample,)
        logpsi, vjp_fn = jax.vjp(raveled_logp, params_flat)
        logpsi_, jvp_fn = jax.linearize(raveled_logp, params_flat)
        assert logpsi.ndim == 1
        # fvp function
        def fisher_apply(x): # (damped) fisher vector product
            # x has the same shape as grad (raveled)
            jvp = jvp_fn(x.conj()) # shape = (n_sample,) same as logp
            mean_jvp = paxis.psum(jnp.sum(jvp * rel_w, axis=0))
            jvp = jvp - shift_factor * mean_jvp
            fvp_local, = vjp_fn(jvp.conj() * rel_w)
            fvp = paxis.psum(fvp_local.real) # local sum is done by vjp
            return fvp + damping * x
        # precond_grads_flat: n_params
        precond_grads_flat, new_solver_state = _stateful_iter_solve(
            fisher_apply, grads_flat, solver_state,
            solver=solver, M=None, **solver_kwargs)
        return precond_grads_flat, new_solver_state

    return GradientTransformationExtraArgs(init_fn, update_fn)


def fisher_inv_direct(score, vec, damping, state=None, *, paxis=None):
    paxis = PmapAxis(paxis) if isinstance(paxis, (str, type(None))) else paxis
    # score: n_sample x n_params
    fisher = (score.T @ score.conj()).real
    fisher = paxis.psum(fisher)
    fisher += damping * jnp.eye(fisher.shape[0])
    return jsp.linalg.solve(fisher, vec, assume_a="pos"), state


def fisher_inv_svd(score, vec, damping, state=None, *, paxis=None):
    # score: n_sample (n) x n_params (m)
    score = _collect_score(score, paxis)
    # vh: n x m
    _, s, vh = fast_svd(score)
    s2inv = 1. / (s**2 + damping)
    v = (jnp.einsum("ia,a,aj,j->i", vh.T, s2inv, vh, vec)
         + 1./damping * (vec - vh.T.dot(vh.dot(vec))))
    return v, state


def fisher_inv_qr(score, vec, damping, state=None, *, paxis=None,
                  all_to_all=True):
    if all_to_all and paxis is not None and paxis.size() > 1:
        return fisher_inv_qr_a2a(score, vec, damping, state, paxis=paxis)
    # score: n_sample (n) x n_params (m)
    score = _collect_score(score, paxis)
    # q: m x n, r: n x n
    q, r = chol_qr(score.T, shift=damping)
    # v ~= (S @ S.T + damping * jnp.eye(m))^-1 @ vec
    v = 1./damping * (vec - q @ (q.T @ vec))
    return v, state


def fisher_inv_qr_a2a(score, vec, damping, state=None, *, paxis):
    assert paxis is not None, "require paxis to perform all_to_all in qr"
    paxis = PmapAxis(paxis) if isinstance(paxis, str) else paxis
    # raw score shape: n_sample (n, loc) x n_params (m)
    nb_loc, np = score.shape # nb is n, np is m
    # pad and all_to_all score, nd: device count, npad: padding length
    score, nd, npad = _collect_score_a2a(score, paxis)
    # q: m (loc) x n, r: n x n
    q, r = chol_qr(score.swapaxes(-1, -2), shift=damping, psum_axis=paxis.name)
    # below is the same as the following commented code
    # q = lax.all_to_all(q, paxis.name, 1, 0, tiled=True)[:np]
    # qqtv = paxis.psum(q @ (q.T @ vec))
    # split vec into local parts
    vec_p = jnp.pad(vec, ((0, npad),), mode="constant")
    vec_l = vec_p.reshape(nd, -1)[lax.axis_index(paxis.name)]
    # calc q @ q.T @ vec considering pmap
    qqtv = q @ paxis.psum(q.T @ vec_l) # shape: m (loc)
    qqtv = paxis.all_gather(qqtv, axis=0, tiled=True)[:np]
    # v ~= (S @ S.T + damping * jnp.eye(m))^-1 @ vec
    v = 1./damping * (vec - qqtv)
    return v, state


def fisher_inv_chol(score, vec, damping, state=None, *, paxis=None,
                    all_to_all=True):
    if all_to_all and paxis is not None and paxis.size() > 1:
        return fisher_inv_chol_a2a(score, vec, damping, state, paxis=paxis)
    # score: n_sample (n) x n_params (m)
    score = _collect_score(score, paxis)
    # w: n x n
    w = score @ score.T + damping * jnp.eye(score.shape[0], dtype=score.dtype)
    # basically q @ q.T @ vec in qr version
    t = score @ vec
    t = jsp.linalg.solve(w, t, assume_a="pos")
    t = score.T @ t
    # v ~= (S @ S.T + damping * jnp.eye(m))^-1 @ vec
    v = 1./damping * (vec - t)
    return v, state


def fisher_inv_chol_a2a(score, vec, damping, state=None, *, paxis):
    assert paxis is not None, "require paxis to perform all_to_all in qr"
    paxis = PmapAxis(paxis) if isinstance(paxis, str) else paxis
    # raw score shape: n_sample (n, loc) x n_params (m)
    nb_loc, np = score.shape # nb is n, np is m
    # pad and all_to_all score, nd: device count, npad: padding length
    score, nd, npad = _collect_score_a2a(score, paxis) # n x m_loc
    # split vec into local parts, vec_l: m_loc
    vec_p = jnp.pad(vec, ((0, npad),), mode="constant")
    vec_l = vec_p.reshape(nd, -1)[lax.axis_index(paxis.name)]
    # concat score.T and vec to do mat mul and psum at once
    st_n_v = jnp.concatenate([score.T, vec_l[:, None]], axis=1)
    w_n_sv = paxis.psum(score @ st_n_v) # n x (n + 1)
    # split w_n_sv into w and sv, w: n x n, sv: n
    w, sv = w_n_sv[:, :-1], w_n_sv[:, -1]
    w += damping * jnp.eye(w.shape[0], dtype=w.dtype)
    # calculate S^T @ W^-1 @ S @ vec considering pmap
    t = jsp.linalg.solve(w, sv, assume_a="pos") # shape: n
    t = score.T @ t # shape: m (loc)
    t = paxis.all_gather(t, axis=0, tiled=True)[:np]
    # v ~= (S @ S.T + damping * jnp.eye(m))^-1 @ vec
    v = 1./damping * (vec - t)
    return v, state


def fisher_inv_iter(score, vec, damping, state, *, paxis=None,
                    solver='cg', precondition=False,
                    tol=1e-10, maxiter=100, x0_mixing=0.):
    paxis = PmapAxis(paxis) if isinstance(paxis, (str, type(None))) else paxis
    # score: n_sample (n) x n_params (m)
    def fisher_apply(x):
        fvp = (score.T.conj() @ (score @ x)).real
        fvp = paxis.psum(fvp)
        return fvp + damping * x
    # M is the preconditioning matrix
    M = None
    if precondition:
        fisher_diag = (score.conj() * score).real.sum(0)
        fisher_diag = paxis.psum(fisher_diag)
        M = lambda x: x / (fisher_diag + damping)
    return _stateful_iter_solve(fisher_apply, vec, state,
                solver=solver, tol=tol, maxiter=maxiter, x0_mixing=x0_mixing, M=M)


def _collect_score(score, paxis):
    paxis = PmapAxis(paxis) if isinstance(paxis, (str, type(None))) else paxis
    # tile real and imag part for complex
    if jnp.iscomplexobj(score):
        score = jnp.concatenate([score.real, score.imag], axis=0)
    # gather all samples together
    score = paxis.all_gather(score, axis=0, tiled=True)
    return score


def _collect_score_a2a(score, paxis):
    paxis = PmapAxis(paxis) if isinstance(paxis, (str, type(None))) else paxis
    # pad score to be devideable by number of devices
    nd = paxis.size()
    npad = (nd - score.shape[-1] % nd) % nd
    score = jnp.pad(score, ((0, 0), (0, npad)), mode="constant")
    # stack real and imag part because complex all_to_all is buggy
    if jnp.iscomplexobj(score):
        score = jnp.stack([score.real, score.imag], axis=0)
    # all to all transpose, split n_params, concat n_sample
    ixp, ixb = score.ndim - 1, score.ndim - 2
    score = lax.all_to_all(score, paxis.name, ixp, ixb, tiled=True)
    score = score.reshape(-1, score.shape[-1]) # flatten the complex dim
    return score, nd, npad


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
