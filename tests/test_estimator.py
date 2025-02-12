# many test functions are borrowed from vmcnet

import jax
import numpy as np
import pytest
import chex
from jax import numpy as jnp

from fvmc.estimator import (build_eval_local_elec, build_eval_local_full,
                            build_eval_total, build_eval_total_weighted)

from .test_hamiltonian import make_test_ions, make_test_log_f, make_test_x


class Dummy:
    pass


def get_sign_log(func, dummy_r=False):
    def log_sign(params, x):
        if isinstance(x, tuple):
            x = x[0]
        f = func(params, x)
        return jnp.sign(f), jnp.log(jnp.abs(f))
    if dummy_r:
        return lambda p, r, x: log_sign(p, x)
    else:
        return log_sign


def get_dummy_apply_pauli(func):
    def apply_fn(params, x, method=None):
        if method is not None and "pauli" in method:
            return jnp.ones((x[0].shape[0], 3))
        return func(params, x[0])
    return apply_fn


def make_dummy_model(apply_fn):
    model = Dummy()
    model.apply = apply_fn
    return model


@pytest.mark.parametrize("pbc", [False, True])
def test_eval_local_shape(pbc):
    f, _ = make_test_log_f()
    model = make_dummy_model(get_sign_log(f))
    nuclei, elems = make_test_ions()
    ndim = nuclei.shape[-1]
    cell = jnp.eye(ndim) * 10 if pbc else None
    eval_local = build_eval_local_elec(model, elems, nuclei, cell)

    a = None
    x = make_test_x()
    eloc, sign, logf, extras = eval_local(a, x)
    assert eloc.shape == sign.shape == logf.shape == tuple()

    bx = jnp.stack([x, x, x], 0) #batch dim has size 3
    beloc, bsign, blogf, bextras = jax.vmap(eval_local, (None, 0))(a, bx)
    assert beloc.shape == bsign.shape == blogf.shape == (3,)


@pytest.mark.parametrize("pbc", [False, True])
def test_eval_local_full_shape(pbc):
    f, _ = make_test_log_f()
    model = make_dummy_model(get_sign_log(f, dummy_r=True))
    r, elems = make_test_ions()
    ndim = r.shape[-1]
    cell = jnp.eye(ndim) * 10 if pbc else None
    eval_local = build_eval_local_full(model, elems, cell)

    a = None
    x = make_test_x()
    eloc, sign, logf, extras = eval_local(a, (r,x))
    assert eloc.shape == sign.shape == logf.shape == tuple()

    br = jnp.stack([r, r, r], 0) #batch dim has size 3
    bx = jnp.stack([x, x, x], 0) #batch dim has size 3
    beloc, bsign, blogf, bextras = jax.vmap(eval_local, (None, 0))(a, (br, bx))
    assert beloc.shape == bsign.shape == blogf.shape == (3,)


def test_eval_local_custom():
    f, _ = make_test_log_f()
    model = make_dummy_model(get_sign_log(f))
    nuclei, elems = make_test_ions()
    dummy_kwargs = {"ke_kwargs": lambda *a,**k: 0.,
                    "pe_kwargs": lambda *a,**k: 0.}
    eval_local = build_eval_local_elec(model, elems, nuclei, **dummy_kwargs)

    a = None
    x = make_test_x()
    eloc, sign, logf, extras = eval_local(a, x)
    assert eloc.shape == sign.shape == logf.shape == tuple()
    chex.assert_trees_all_equal(eloc, extras["e_kin"], extras["e_coul"], 0.)

    bx = jnp.stack([x, x, x], 0) #batch dim has size 3
    beloc, bsign, blogf, bextras = jax.vmap(eval_local, (None, 0))(a, bx)
    assert beloc.shape == bsign.shape == blogf.shape == (3,)
    chex.assert_trees_all_equal(beloc, bextras["e_kin"], bextras["e_coul"], jnp.zeros(3))


def test_eval_local_full_custom():
    f, _ = make_test_log_f()
    model = make_dummy_model(get_sign_log(f, dummy_r=True))
    r, elems = make_test_ions()
    dummy_kwargs = {"ke_kwargs": lambda *a,**k: 0.,
                    "pe_kwargs": lambda *a,**k: 0.}
    eval_local = build_eval_local_full(model, elems, **dummy_kwargs)

    a = None
    x = make_test_x()
    eloc, sign, logf, extras = eval_local(a, (r,x))
    assert eloc.shape == sign.shape == logf.shape == tuple()
    chex.assert_trees_all_equal(eloc, extras["e_kin"], extras["e_coul"], 0.)

    br = jnp.stack([r, r, r], 0) #batch dim has size 3
    bx = jnp.stack([x, x, x], 0) #batch dim has size 3
    beloc, bsign, blogf, bextras = jax.vmap(eval_local, (None, 0))(a, (br, bx))
    assert beloc.shape == bsign.shape == blogf.shape == (3,)
    chex.assert_trees_all_equal(beloc, bextras["e_kin"], bextras["e_coul"], jnp.zeros(3))


def test_eval_local_spin():
    f, _ = make_test_log_f()
    model = make_dummy_model(get_dummy_apply_pauli(get_sign_log(f)))
    nuclei, elems = make_test_ions()
    spin_pots = {"ones": lambda x: jnp.ones((x.shape[0], 3)),
                 "one1b": lambda x: jnp.ones(3)}
    eval_local = build_eval_local_elec(model, elems, nuclei, spin_pots=spin_pots)

    a = None
    x = make_test_x()
    n_elec = x.shape[0]
    s = jnp.zeros(n_elec)
    eloc, sign, logf, extras = eval_local(a, (x, s))
    assert eloc.shape == sign.shape == logf.shape == tuple()
    target_ke = -0.5 * (x.shape[-1] * x.shape[-2]) * 2 / f(None, x)
    np.testing.assert_allclose(extras["e_kin"], target_ke)
    chex.assert_trees_all_close(extras["e_ones"], extras["e_one1b"], n_elec * 3)


def test_eval_local_full_spin():
    f, _ = make_test_log_f()
    model = make_dummy_model(get_sign_log(f, dummy_r=True))
    r, elems = make_test_ions()
    eval_local = build_eval_local_full(model, elems)

    a = None
    x = make_test_x()
    n_elec = x.shape[0]
    s = jnp.zeros(n_elec)
    eloc, sign, logf, extras = eval_local(a, (r, (x, s)))
    assert eloc.shape == sign.shape == logf.shape == tuple()
    target_ke = -0.5 * (x.shape[-1] * x.shape[-2]) * 2 / f(None, x)
    np.testing.assert_allclose(extras["e_kin"], target_ke)


@pytest.mark.parametrize("mini_batch", [None, 1])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("clipping", [None, 0.75])
def test_eval_total(clipping, weighted, mini_batch):
    log_psi = lambda a, x: (jnp.sign(x.mean(-1)-2), a * jnp.sum(jnp.square(x), axis=(-1)))
    model = make_dummy_model(log_psi)

    def local_energy(x):
        return 4.5 - x.mean(-1)

    def eval_local(params, x):
        eloc = local_energy(x)
        sign, logf = model.apply(params, x)
        return eloc, sign, logf, {}

    a = 3.5
    x = make_test_x()
    x = x + jax.random.normal(jax.random.PRNGKey(0), x.shape)

    log_sample = 2. * log_psi(a, x)[1]

    # log_psi_grad_a = jnp.array([5.0, 25.0, 61.0])
    log_psi_grad_a = jnp.sum(jnp.square(x), axis=(-1))
    target_local_energies = local_energy(x)
    target_energy = target_local_energies.mean()
    target_variance = target_local_energies.var()
    tv = jnp.abs(target_local_energies - target_energy).mean()
    clipped_local_energies = (jnp.clip(target_local_energies,
        target_energy-clipping*tv, target_energy+clipping*tv)
        if clipping and clipping > 0 else target_local_energies)
    target_grad_energy = 2.0 * jnp.mean(
        (clipped_local_energies - target_energy) * log_psi_grad_a
    )

    eval_total = build_eval_total(eval_local, clipping,
                    center_shifting=False, mini_batch=mini_batch,
                    use_weighted=weighted)
    eval_total_grad = jax.value_and_grad(eval_total, has_aux=True)

    # loss, aux = eval_total(a, (x, log_sample))
    (loss, aux), grad_energy = eval_total_grad(a, (x, log_sample))
    energy = aux["e_tot"]
    variance = aux["var_e"]

    np.testing.assert_allclose(energy, target_energy)
    np.testing.assert_allclose(variance, target_variance)
    np.testing.assert_allclose(grad_energy, target_grad_energy, rtol=1e-6)
