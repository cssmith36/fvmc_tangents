import jax
from jax import lax
from jax import numpy as jnp

from .utils import pdist, cdist


def calc_coulomb(pos, charge):
    n = pos.shape[-2]
    dist = pdist(pos)
    charge =jnp.atleast_1d(charge)
    cmat = jnp.expand_dims(charge, -1) * jnp.expand_dims(charge, -2)
    coulomb = jnp.triu(cmat / (dist + jnp.eye(n)), k=1)
    return coulomb.sum((-1,-2))


def calc_coulomb_2(pos_a, charge_a, pos_b, charge_b):
    dist = cdist(pos_a, pos_b)
    charge_a, charge_b = map(jnp.atleast_1d, [charge_a, charge_b])
    cmat = jnp.expand_dims(charge_a, -1) * jnp.expand_dims(charge_b, -2)
    coulomb = cmat / dist
    return coulomb.sum((-1,-2))


def calc_potential_energy(ions, charges, x):
    # x is electron positions
    el_el = calc_coulomb(x, -1)
    el_ion = calc_coulomb_2(x, -1, ions, charges)
    ion_ion = calc_coulomb(ions, charges)
    return el_el + el_ion + ion_ion


def calc_kinetic_energy(log_psi, params, x):
    # adapted from FermiNet and vmcnet
    # calc -0.5 * (\nable^2 \psi) / \psi
    # handle batch of x automatically

    def _lapl_over_psi(x):
        # (\nable^2 f) / f = \nabla^2 log|f| + (\nabla log|f|)^2
        # x is assumed to have shape [n_ele, 3], not batched
        x_shape = x.shape
        flat_x = x.reshape(-1)
        ncoord = flat_x.size

        f = lambda flat_x: log_psi(params, flat_x.reshape(x_shape)) # take flattened x
        grad_f = jax.grad(f)
        grad_value, dgrad_f = jax.linearize(grad_f, flat_x)

        eye = jnp.eye(ncoord)
        loop_fn = lambda i, val: val + dgrad_f(eye[i])[i]
        laplacian = (grad_value**2).sum() + lax.fori_loop(0, ncoord, loop_fn, 0.0)
        return laplacian
    
    if x.ndim == 2:
        lapl_fn = _lapl_over_psi
    elif x.ndim == 3:
        lapl_fn = jax.vmap(_lapl_over_psi)
    else:
        raise ValueError("only support x with ndim being 2 or 3")
    
    return -0.5 * lapl_fn(x)


def calc_local_energy(log_psi, params, ions, charges, x):
    ke = calc_kinetic_energy(log_psi, params, x) 
    pe = calc_potential_energy(ions, charges, x)
    return ke + pe
    