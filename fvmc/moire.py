import jax.numpy as jnp
from .utils import Array


class OneShell:
    r"""Smoothest approximation of the moire potential defined over the
      first shell of reciprocal-space lattice
    Ref: F. Wu, T. Lovorn, A.H. MacDonald, PRL 118, 147401 (2017) eq. (1)
      pair exp(i x) + exp(-i x) to 2cos(x)

    .. math::
      V(r) = V_m * \sum\limits_{j=1}^3 2cos(G_j \cdot r + \phi)
    :math:`G_j` are reciprocal lattice vectors of the moire unit cell
    """

    def __init__(self, cell: Array, am_length: float, vm_depth: float, phi_shape: float):
        """create moire reciprocal lattice vectors (kvecs)
          currently only triangular moire unit cell with lattice constant am_length

        Args:
            cell (Array): simulation cell
            am_length (float): moire unit cell lattice constant
            vm_depth (float): depth of moire potential
            phi_shape (float): shape of moire potential
        """
        # !!!! hard-code triangular lattice
        bm = 4*jnp.pi/3**0.5/am_length
        recvec_moire = bm*jnp.array([
            [3**0.5/2, 0.5],
            [0, 1],
        ])
        # check supercell is commensurate
        legal_supercell =  _check_recvec(cell, recvec_moire)
        if not legal_supercell:
            cell_moire = 2*jnp.pi*jnp.linalg.inv(recvec_moire).T
            msg = 'supercell \n%s\n is not commensurate' % str(cell)
            msg += ' with moire unit cell \n%s' % str(cell_moire)
            raise RuntimeError(msg)
        kidx = jnp.array([
            [0, 1],
            [-1, 0],
            [1, -1],
        ])
        self.kvecs = kidx @ recvec_moire
        self.vm = vm_depth
        self.phi = phi_shape

    def __call__(self, x: Array) -> Array:
        kr = jnp.tensordot(self.kvecs, x, (-1, -1))  # (n_k, n_pts)
        cos_term = jnp.cos(kr+self.phi).sum(axis=0)
        return self.vm*2.0*cos_term

    def calc_pe(self, x: Array) -> float:
        """calculate potential energy of electrons in moire potential"""
        vals = self(x)
        return vals.sum()


def _check_recvec(cell: Array, recvec_moire: Array):
    recvec_sup = 2*jnp.pi*jnp.linalg.inv(cell).T
    #   put moire recvec in fractional units of supercell recvec
    g_candidates = recvec_moire @ jnp.linalg.inv(recvec_sup)
    g_idx = g_candidates.round().astype(int)
    return jnp.allclose(g_idx, g_candidates)
