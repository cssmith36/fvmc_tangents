import jax
from jax import lax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from .utils import pdist, cdist, adaptive_grad


def calc_coulomb(charge, pos):
    n = pos.shape[-2]
    dist = pdist(pos)
    charge =jnp.atleast_1d(charge)
    cmat = jnp.expand_dims(charge, -1) * jnp.expand_dims(charge, -2)
    coulomb = jnp.triu(cmat / (dist + jnp.eye(n)), k=1)
    return coulomb.sum((-1,-2))


def calc_coulomb_2(charge_a, pos_a, charge_b, pos_b):
    dist = cdist(pos_a, pos_b)
    charge_a, charge_b = map(jnp.atleast_1d, [charge_a, charge_b])
    cmat = jnp.expand_dims(charge_a, -1) * jnp.expand_dims(charge_b, -2)
    coulomb = cmat / dist
    return coulomb.sum((-1,-2))


def calc_pe(elems, r, x):
    # r is nuclei position
    # x is electron positions
    el_el = calc_coulomb(-1, x)
    el_ion = calc_coulomb_2(-1, x, elems, r)
    ion_ion = calc_coulomb(elems, r)
    return el_el + el_ion + ion_ion


def calc_ke_elec(log_psi, x):
    # adapted from FermiNet and vmcnet
    # calc -0.5 * (\nable^2 \psi) / \psi
    # handle batch of x automatically

    def _lapl_over_psi(x):
        # (\nable^2 f) / f = \nabla^2 log|f| + (\nabla log|f|)^2
        # x is assumed to have shape [n_ele, 3], not batched
        x_shape = x.shape
        flat_x = x.reshape(-1)
        ncoord = flat_x.size

        f = lambda flat_x: log_psi(flat_x.reshape(x_shape)) # take flattened x
        grad_f = adaptive_grad(f)
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
        raise ValueError(f"only support x with ndim equals 2 or 3, get {x.ndim}")
    
    return -0.5 * lapl_fn(x)


def get_nuclei_mass(elems):
    from .utils import PROTON_MASS, ISOTOPE_MAIN
    # neutrons are treated as 1
    mass = PROTON_MASS * jnp.asarray(ISOTOPE_MAIN)[elems.astype(int)]
    return mass


def calc_ke_full(log_psi, mass, r, x):
    # adapted from FermiNet and vmcnet
    # calc -0.5 * (\nable^2 \psi) / \psi
    # handle batch of r, x automatically
    mass = jnp.reshape(mass, -1)

    def _lapl_over_psi(r, x):
        # (\nable^2 f) / f = \nabla^2 log|f| + (\nabla log|f|)^2
        # r and x is assumed to have shape [n_ion/elec, 3], not batched
        flat_in, unravel = ravel_pytree((r, x))
        ncoord = flat_in.size
        assert mass.size == r.shape[0]
        
        f = lambda flat_in: log_psi(*unravel(flat_in)) # take flattened x
        grad_f = adaptive_grad(f)
        grad_value, dgrad_f = jax.linearize(grad_f, flat_in)

        minv = jnp.concatenate([jnp.repeat(1/mass, r.shape[1]), 
                                jnp.ones(x.size)])
        minv_mat = jnp.diag(minv)
        loop_fn = lambda i, val: val + dgrad_f(minv_mat[i])[i]
        laplacian = ((minv * grad_value**2).sum() 
                     + lax.fori_loop(0, ncoord, loop_fn, 0.0))
        return laplacian
    
    if r.ndim == x.ndim == 2:
        lapl_fn = _lapl_over_psi
    elif r.ndim == x.ndim == 3:
        lapl_fn = jax.vmap(_lapl_over_psi)
    else:
        raise ValueError(f"unsupported r.ndim: {r.ndim} and x.ndim: {x.ndim}")
    
    return -0.5 * lapl_fn(r, x)


def calc_local_energy(log_psi, elems, r, x, cell=None, nuclei_ke=False):
    # do not use this one directly in QMC with a fixed cell. 
    # It's slow as it will rebuild the g points every time.
    if nuclei_ke:
        mass = get_nuclei_mass(elems)
        ke = calc_ke_full(log_psi, mass, r, x)
    else:
        ke = calc_ke_elec(log_psi, x) 
    if cell is not None:
        from .ewaldsum import EwaldSum
        pe = EwaldSum(cell).calc_pe(elems, r, x)
    else:
        pe = calc_pe(elems, r, x)
    return ke + pe
    