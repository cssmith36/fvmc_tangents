# many test functions are borrowed from vmcnet

import pytest
import jax
import numpy as np
from jax import numpy as jnp

from vdmc.utils import pdist
from vdmc.wavefunction import Jastrow, SimpleOrbital, Slater, make_jastrow_slater


def make_collapse_conf():
    ion_charges = jnp.array([1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
    ion_pos = jnp.array(
        [
            [0.0, 0.0, -2.5],
            [0.0, 0.0, -1.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 2.5],
        ]
    )
    elec_pos = jnp.expand_dims(
        jnp.array(
            [
                [0.0, 0.0, -2.5],
                [0.0, 0.0, -1.5],
                [0.0, 0.0, -1.5],
                [0.0, 0.0, -0.5],
                [0.0, 0.0, -0.5],
                [0.0, 0.0, -0.5],
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 1.5],
                [0.0, 0.0, 2.5],
            ]
        ),
        axis=0,
    )
    return ion_pos, ion_charges, elec_pos


_key0 = jax.random.PRNGKey(0)


def test_jastrow():
    ions, charges, x = make_collapse_conf()
    jastrow = Jastrow(ions, charges)
    params = jastrow.init(_key0, x)
    
    actual_out = jastrow.apply(params, x)
    assert actual_out.shape == (1,)
    np.testing.assert_allclose(actual_out[0], 0.)

    new_x = x.at[0,0,2].set(2e10)[0, :-2, :] # now remve batch
    new_out = jastrow.apply(params, new_x)
    np.testing.assert_allclose(new_out, -3 * 2e10)


def test_orbital_shape():
    ions, charges, x = make_collapse_conf()
    n_batch, n_el = x.shape[:-1]
    n_orb = 7
    orbital = SimpleOrbital(ions, n_orb, n_hidden=1)
    params = orbital.init(_key0, x)

    assert orbital.apply(params, x).shape == (n_batch, n_el, n_orb)
    assert orbital.apply(params, x[0]).shape == (n_el, n_orb)


@pytest.mark.parametrize("full_det,spin", [(True, None), (False, 1)])
def test_slater_antisymm(full_det, spin):
    ions, charges, x = make_collapse_conf()
    n_batch, n_el = x.shape[:-1]
    slater = Slater(ions, charges, full_det=full_det, orbital_args={"n_hidden": 1})
    params = slater.init(_key0, x)

    iperm = jnp.arange(n_el, dtype=int).at[:2].set([1,0])
    px = x[:, iperm, :]
    
    sign1, logf1 = slater.apply(params, x)
    sign2, logf2 = slater.apply(params, px)
    assert sign1.shape == logf1.shape == (n_batch,)
    np.testing.assert_allclose(sign1, -sign2)
    np.testing.assert_allclose(logf1, logf2)


def test_jastrow_slater():
    ions, charges, x = make_collapse_conf()
    x = x[0]
    n_el = x.shape[0]
    model = make_jastrow_slater(ions, charges, None, full_det=True, orbital_args={"n_hidden": 1})
    params = model.init(_key0, x)
    subp0 = {"params": params["params"]["submodels_0"]}
    subp1 = {"params": params["params"]["submodels_1"]}

    sign, logf = model.apply(params, x)
    log_jas = model.submodels[0].apply(subp0, x)
    sign_sla, log_sla = model.submodels[1].apply(subp1, x)
    np.testing.assert_allclose(sign, sign_sla)
    np.testing.assert_allclose(logf, log_jas + log_sla)
