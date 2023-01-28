# many test functions are borrowed from vmcnet

import pytest
import jax
import numpy as np
from jax import numpy as jnp

from vdmc.estimator import build_eval_local, build_eval_total
from .test_hamiltonian import make_test_log_f, make_test_ions, make_test_x


class Dummy:
    pass


def get_sign_log(func):
    def log_sign(params, x):
        f = func(params, x)
        return jnp.sign(f), jnp.log(jnp.abs(f))
    return log_sign


def make_dummy_model(apply_fn):
    model = Dummy()
    model.apply = apply_fn
    return model


def test_eval_local_shape():
    f, logf = make_test_log_f()
    model = make_dummy_model(get_sign_log(f))
    ions, elems = make_test_ions()
    eval_local = build_eval_local(model, ions, elems)
    
    a = None
    x = make_test_x()
    eloc, sign, logf = eval_local(a, x)
    assert eloc.shape == sign.shape == logf.shape == (2,)

    bx = jnp.stack([x, x, x], 0) #batch dim has size 3
    beloc, bsign, blogf = jax.vmap(eval_local, (None, 0))(a, bx)
    assert beloc.shape == bsign.shape == blogf.shape == (3, 2)


@pytest.mark.parametrize("clipping", [0., 0.75])
def test_eval_total(clipping):
    log_psi = lambda a, x: (jnp.sign(x.mean(-1)-2), a * jnp.sum(jnp.square(x), axis=(-1)))
    model = make_dummy_model(log_psi)

    def eval_local(params, x):
        eloc = 4.5 - x.mean(-1)
        sign, logf = model.apply(params, x)
        return jnp.array([eloc, eloc]), jnp.array([sign, sign]), jnp.array([logf, logf])

    a = 3.5
    x = make_test_x()

    log_sample = 2. * log_psi(a, x)[1]

    log_psi_grad_x = jnp.array([5.0, 25.0, 61.0])
    target_local_energies = jnp.array([3.0, 1.0, -1.0])
    target_energy = target_local_energies.mean()
    target_variance = target_local_energies.var()
    tv = jnp.abs(target_local_energies - target_energy).mean()
    clipped_local_energies = (jnp.clip(target_local_energies, 
        target_energy-clipping*tv, target_energy+clipping*tv)
        if clipping > 0 else target_local_energies)
    target_grad_energy = 2.0 * jnp.mean(
        (clipped_local_energies - target_energy) * log_psi_grad_x
    )

    eval_total = build_eval_total(eval_local, clipping,)
    eval_total_grad = jax.value_and_grad(eval_total, has_aux=True)

    # loss, aux = eval_total(a, (x, log_sample))
    (loss, aux), grad_energy = eval_total_grad(a, (x, log_sample))
    energy = aux["e_tot"]
    variance = aux["var_e"]

    np.testing.assert_allclose(energy, target_energy)
    np.testing.assert_allclose(variance, target_variance)
    np.testing.assert_allclose(grad_energy, target_grad_energy, rtol=1e-6)
