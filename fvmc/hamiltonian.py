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


def laplacian_over_f(log_f, scale=None, forward_mode=False, partition_size=1):
    # (\nable^2 f) / f = \nabla^2 log|f| + (\nabla log|f|)^2
    if forward_mode:
        import fwdlap
    scale = ravel_pytree(scale)[0] if scale is not None else jnp.array([1.0])

    # normal bwd-fwd hessian version
    def _lapl_backward(*args):
        flat_in, unravel = ravel_pytree(args)
        ncoord = flat_in.size
        seye = jnp.eye(ncoord) * jnp.diag(scale)
        f = lambda flat_in: log_f(*unravel(flat_in)) # take flattened x
        grad_f = adaptive_grad(f)
        grad_value, dgrad_f = jax.linearize(grad_f, flat_in)
        laplacian = (scale * grad_value**2).sum()
        # different partition size need different implementation
        if partition_size is None:
            hess = jax.vmap(dgrad_f)(seye)
            laplacian += hess.trace()
        elif partition_size == 1:
            loop_fn = lambda i, val: val + dgrad_f(seye[i])[i]
            laplacian += lax.fori_loop(0, ncoord, loop_fn, 0.0)
        else:
            seye = seye.reshape(-1, partition_size, ncoord)
            hess = lax.map(jax.vmap(dgrad_f), seye)
            laplacian += hess.reshape(ncoord, ncoord).trace()
        return laplacian

    # fwd laplacian version
    def _lapl_forward(*args):
        flat_in, unravel = ravel_pytree(args)
        ncoord = flat_in.size
        seye = jnp.eye(ncoord) * jnp.diag(jnp.sqrt(scale))
        zero = fwdlap.Zero.from_value(flat_in)
        f = lambda flat_in: log_f(*unravel(flat_in)) # take flattened x
        if partition_size is None:
            primals, grads, laps = fwdlap.lap(f, (flat_in,), (seye,), (zero,))
            laplacian = (grads**2).sum() + laps
        else:
            seye = seye.reshape(-1, partition_size, ncoord)
            primals, f_lap_pe = fwdlap.lap_partial(f, (flat_in,), (seye[0],), (zero,))
            def loop_fn(i, val):
                (jac, lap) = f_lap_pe((seye[i],), (zero,))
                return val + (jac**2).sum() + lap
            laplacian = lax.fori_loop(0, ncoord//partition_size, loop_fn, 0.0)
        return laplacian

    return _lapl_forward if forward_mode else _lapl_backward


def calc_ke_elec(log_psi, x, *, forward_mode=False, partition_size=1):
    # calc -0.5 * (\nable^2 \psi) / \psi
    # handle batch of x automatically
    lapl_fn = laplacian_over_f(
        log_psi,
        scale=None,
        forward_mode=forward_mode,
        partition_size=partition_size)
    return -0.5 * lapl_fn(x)


def calc_ke_full(log_psi, mass, r, x, *, forward_mode=False, partition_size=1):
    # calc -0.5 * (\nable^2 \psi) / \psi
    # handle batch of r, x automatically
    mass = jnp.reshape(mass, -1)
    minv = (jnp.repeat(1/mass, r.shape[-1]), jnp.ones(x.shape[-1] * x.shape[-2]))
    lapl_fn = laplacian_over_f(
        log_psi,
        scale=minv,
        forward_mode=forward_mode,
        partition_size=partition_size)
    return -0.5 * lapl_fn(r, x)


def get_nuclei_mass(elems):
    from .utils import PROTON_MASS, ISOTOPE_MAIN
    # neutrons are treated as 1
    mass = PROTON_MASS * jnp.asarray(ISOTOPE_MAIN)[elems.astype(int)]
    return mass


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
