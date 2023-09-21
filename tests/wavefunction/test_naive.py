# many test functions are borrowed from vmcnet

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from fvmc.utils import pdist
from fvmc.wavefunction.base import FixNuclei
from fvmc.wavefunction.naive import (NucleiGaussian, NucleiGaussianSlater,
                                     SimpleJastrow, SimpleOrbital,
                                     SimpleSlater, build_jastrow_slater)


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
    nuclei, elems, x = make_collapse_conf()
    jastrow = SimpleJastrow(elems)
    params = jastrow.init(_key0, nuclei, x)

    actual_out = jastrow.apply(params, nuclei, x)
    assert actual_out.shape == (1,)
    np.testing.assert_allclose(actual_out[0], 0.)

    new_x = x.at[0,0,2].set(2e10)[0, :-2, :] # now remve batch
    new_out = jastrow.apply(params, nuclei, new_x)
    np.testing.assert_allclose(new_out, -3 * 2e10, rtol=1e-6)


def test_orbital_shape():
    nuclei, elems, x = make_collapse_conf()
    n_batch, n_el = x.shape[:-1]
    n_orb = 7
    orbital = SimpleOrbital(n_orb, n_hidden=1)
    params = orbital.init(_key0, nuclei, x)

    assert orbital.apply(params, nuclei, x).shape == (n_batch, n_el, n_orb)
    assert orbital.apply(params, nuclei, x[0]).shape == (n_el, n_orb)


@pytest.mark.slow
@pytest.mark.parametrize("full_det,spin", [(True, None), (False, 1)])
def test_slater_antisymm(full_det, spin):
    nuclei, elems, x = make_collapse_conf()
    n_batch, n_el = x.shape[:-1]
    spins = (n_el // 2, n_el - n_el // 2)
    slater = SimpleSlater(spins=spins, full_det=full_det,
                          orbital_args={"n_hidden": 1})
    params = slater.init(_key0, nuclei, x)

    x = x + jax.random.normal(_key0, x.shape)
    iperm = jnp.arange(n_el, dtype=int).at[:2].set([1,0])
    px = x[:, iperm, :]

    sign1, logf1 = slater.apply(params, nuclei, x)
    sign2, logf2 = slater.apply(params, nuclei, px)
    assert sign1.shape == logf1.shape == (n_batch,)
    np.testing.assert_allclose(sign1, -sign2)
    np.testing.assert_allclose(logf1, logf2)


@pytest.mark.slow
def test_jastrow_slater():
    nuclei, elems, x = make_collapse_conf()
    x = x[0]
    n_el = x.shape[0]
    spins = (n_el // 2, n_el - n_el // 2)
    model = FixNuclei(
        build_jastrow_slater(
            elems, spins, nuclei, full_det=True, orbital_args={"n_hidden": 1}),
        nuclei=nuclei)
    params = model.init(_key0, x)
    subp0 = {"params": params["params"]['model']["submodels_0"]}
    subp1 = {"params": params["params"]['model']["submodels_1"]}

    sign, logf = model.apply(params, x)
    log_jas = model.model.submodels[0].apply(subp0, nuclei, x)
    sign_sla, log_sla = model.model.submodels[1].apply(subp1, nuclei, x)
    np.testing.assert_allclose(sign, sign_sla)
    np.testing.assert_allclose(logf, log_jas + log_sla)


def test_nuclei_gaussian():
    nuclei, elems, x = make_collapse_conf()
    r = nuclei
    model = NucleiGaussian(nuclei, 0.1)
    params = model.init(_key0, r, x)

    sign, logf = model.apply(params, r, x)
    np.testing.assert_allclose(sign, 1.0)
    np.testing.assert_allclose(logf, 0.0)


def test_nuclei_gaussian_slater():
    nuclei, elems, x = make_collapse_conf()
    r = nuclei
    model = NucleiGaussianSlater(nuclei, 0.01)
    params = model.init(_key0, r, x)

    sign, logf = model.apply(params, r, x)
    np.testing.assert_allclose(sign, 1.0)
    np.testing.assert_allclose(logf, 0.0)

    iperm = jnp.arange(r.shape[0], dtype=int).at[:2].set([1,0])
    pr = r[iperm, :]
    psign, plogf = model.apply(params, pr, x)
    np.testing.assert_allclose(psign, -1.0)
    np.testing.assert_allclose(plogf, logf)


