import jax
from jax import numpy as jnp
from flax import linen as nn

from .utils import Array
from .utils import cdist, pdist, fix_init, _t_real


# follow the TwoBodyExpDecay class in vmcnet
class Jastrow(nn.Module):
    """Isotropic exponential decay two-body Jastrow model.
    
    The decay is isotropic in the sense that each electron-nuclei and electron-electron
    term is isotropic, i.e. radially symmetric. The computed interactions are:

        sum_i(-sum_j Z_j ||elec_i - ion_j|| + sum_k Q ||elec_i - elec_k||)

    (no exponential because it we are working with log of wavefunctions.)
    Z_j and Q are parameters that are initialized to be ion charges and 1.
    """

    ions : Array
    charges : Array

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # calculate initial scale, so that it returns 0 if all electrons are on ions
        cmat = jnp.expand_dims(self.charges, -1) * jnp.expand_dims(self.charges, -2)
        scale = 0.5 * jnp.sum(pdist(self.ions) * cmat)
        # make z and q parameters
        z = self.param("z", fix_init, self.charges, _t_real)
        q = self.param("q", fix_init, 1.0, _t_real)
        # distance matrices
        r_ei = cdist(x, self.ions)
        r_ee = pdist(x)
        # interaction terms
        corr_ei = jnp.sum(r_ei * z, axis=-1)
        corr_ee = jnp.sum(jnp.triu(r_ee) * q, axis=-1)
        return jnp.sum(corr_ee - corr_ei, axis=-1) + scale