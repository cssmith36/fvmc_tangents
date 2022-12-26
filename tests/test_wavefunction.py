import jax
import numpy as np
from jax import numpy as jnp

from vdmc.utils import pdist
from vdmc.wavefunction import Jastrow


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


key0 = jax.random.PRNGKey(0)


def test_jastrow():
    ions, charges, x = make_collapse_conf()
    jastrow = Jastrow(ions, charges)
    params = jastrow.init(key0, x)
    
    actual_out = jastrow.apply(params, x)
    assert actual_out.shape == (1,)
    np.testing.assert_allclose(actual_out[0], 0.)

    new_x = x.at[0,0,2].set(2e10)[0, :-2, :] # now remve batch
    new_out = jastrow.apply(params, new_x)
    np.testing.assert_allclose(new_out, -3 * 2e10)
