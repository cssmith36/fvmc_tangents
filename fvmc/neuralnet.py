import functools
from dataclasses import field
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp


from .utils import parse_activation, diffmat, cdist, pdist, adaptive_residual


# for all functions, we use the following convention:
# r for atom positions
# x for electron positions
# elems for the nuclear charges (elements)


class Embedding(nn.Module):
    embedding_size: int
    activation: str

    @nn.compact
    def __call__(self, r, x):
        n_embd = self.embedding_size
        n_elec = x.shape[0]
        n_atoms = r.shape[0]
        actv_fn = parse_activation(self.activation, rescale=True)

        # Electron-atom distances
        d_ei = x[:, None] - r[None]
        d_ei_norm = jnp.linalg.norm(d_ei, keepdims=True, axis=-1)
        d_ei = jnp.concatenate([d_ei, d_ei_norm], axis=-1)
        h1_scaling = jnp.log(1+d_ei_norm)/d_ei_norm
        h1 = d_ei * h1_scaling

        # Electron-electron distances
        d_ee = x[:, None] - x[None]
        d_ee_norm = jnp.linalg.norm(
            d_ee + jnp.eye(n_elec)[..., None],
            keepdims=True,
            axis=-1
        )
        h2_scaling = jnp.log(1+d_ee_norm)/d_ee_norm
        d_ee = jnp.concatenate([d_ee, 
            d_ee_norm * (1.0 - jnp.eye(n_elec)[..., None])], axis=-1)
        h2 = d_ee * h2_scaling

        # Invariant electron-nuclei embedding
        nuc_embedding = self.param(
            'nuc_embedding',
            nn.initializers.normal(1/onp.sqrt(4)),
            (n_atoms, 4, n_embd)
        )
        nuc_bias = self.param(
            'nuc_bias',
            nn.initializers.normal(1.0),
            (n_atoms, n_embd)
        )
        h1 = jnp.einsum('nmi,mio->nmo', h1, nuc_embedding) + nuc_bias
        h1 = actv_fn(h1) # TODO: check if this is better to use layer norm
        h1 = h1.mean(1)
        return h1, h2, d_ei, d_ee


def aggregate_features(h1, h2, n_elec, absolute_spin):
    n_elec = onp.array(n_elec)
    assert n_elec.sum() == h1.shape[0]
    n_up, n_dn = n_elec
    # Single input
    h2_mean = jnp.stack(
        [
            h2_spin.mean(axis=1)
            for h2_spin in jnp.split(h2, n_elec[:1], axis=1) 
            if h2_spin.size > 0
        ], axis=-2)
    if not absolute_spin:
        h2_mean = h2_mean.at[n_up:].set(h2_mean[n_up:, (1, 0)])
    one_in = jnp.concatenate([h1, h2_mean.reshape(h1.shape[0], -1)], axis=-1)
    # Global input
    h1_up, h1_dn = jnp.split(h1, n_elec[:1], axis=0)
    all_up, all_dn = h1_up.mean(0), h1_dn.mean(0)
    if absolute_spin:
        all_in = jnp.array([[all_up, all_dn], [all_up, all_dn]])
    else:
        all_in = jnp.array([[all_up, all_dn], [all_dn, all_up]])
    all_in = all_in.reshape(2, -1)
    return one_in, all_in


class FermiLayer(nn.Module):
    single_size: int
    pair_size: int
    n_elec: Tuple[int, int]
    activation: str
    absolute_spin: bool
    update_pair_independent: bool

    @nn.compact
    def __call__(self, h1, h2):
        actv_fn = parse_activation(self.activation)

        # Single update
        one_in, all_in = aggregate_features(h1, h2, self.n_elec, self.absolute_spin)
        # per electron contribution
        one_new = nn.Dense(self.single_size)(one_in)
        # global contribution
        all_new = nn.Dense(self.single_size, use_bias=False)(all_in) # [2, n_single]
        all_new = all_new.repeat(onp.array(self.n_elec), axis=0) # broadcast to both spins
        # combine both of them to get new h1
        h1_new = (one_new + all_new) #/ jnp.sqrt(2.0)
        h1 = adaptive_residual(h1, actv_fn(h1_new))
        
        # Pairwise update
        if self.update_pair_independent:
            h2_new = nn.Dense(self.pair_size)(h2)
        else:
            u, d = jnp.split(h2, self.n_elec[:1], axis=0)
            uu, ud = jnp.split(u, self.n_elec[:1], axis=1)
            du, dd = jnp.split(d, self.n_elec[:1], axis=1)
            same = nn.Dense(self.pair_size)
            diff = nn.Dense(self.pair_size)
            h2_new = jnp.concatenate([
                jnp.concatenate([same(uu), diff(ud)], axis=1),
                jnp.concatenate([diff(du), same(dd)], axis=1),
            ], axis=0)
        if h2.shape != h2_new.shape: # fitst layer
            h2 = jnp.tanh(h2_new)
        else:
            h2 = adaptive_residual(h2, actv_fn(h2_new))
        return h1, h2


class IsotropicEnvelope(nn.Module):
    charges: Tuple[int, ...]
    out_dim: int
    determinants: int
    sigma_init: float = 1
    pi_init: float = 1

    @nn.compact
    def __call__(self, x):  
        # x is of shape n_elec, n_nuclei, 4
        n_nuclei = x.shape[1]
        def sigma_init(_, shape):
            n_k = onp.arange(1, 9)
            n_k = n_k.repeat(n_k**2)
            charges = onp.array(self.charges)[:, None]
            return jnp.array(charges / n_k[:shape[1]])[..., None].repeat(shape[2], 2)
        sigma = self.param(
            'sigma',
            sigma_init if isinstance(self.sigma_init, str) else nn.initializers.constant(self.sigma_init),
            (n_nuclei, self.out_dim, self.determinants)
        ).reshape(n_nuclei, -1)
        pi = self.param(
            'pi',
            nn.initializers.constant(self.pi_init),
            (n_nuclei, self.out_dim * self.determinants)
        )
        sigma = nn.softplus(sigma)
        pi = nn.softplus(pi)
        return jnp.sum(jnp.exp(-x[..., -1:] * sigma) * pi, axis=1)


class Orbitals(nn.Module):
    spins: Tuple[int, int]
    charges: Tuple[int, ...]
    determinants: int
    full_det: bool
    share_weights: bool = False

    @nn.compact
    def __call__(self, h_one, r_im):
        # h_one is n_elec, D
        # r_im is n_elec, n_atom, 4
        kernel_init = functools.partial(nn.initializers.variance_scaling, mode='fan_in', distribution='truncated_normal')

        # Orbital functions
        def orbital_fn(h, r, out_dim, kernel_scale=1, sigma_init=1, pi_init=1):
            n_param = out_dim * self.determinants
            # Different initialization to ensure zero off diagonals
            if isinstance(kernel_scale, slice):
                dense = nn.Dense(
                    n_param,
                    kernel_init=lambda key, shape, dtype=jnp.float_: jnp.concatenate([
                        kernel_init(1.0)(key, shape=(*shape[:-1], self.determinants*self.spins[0]), dtype=dtype),
                        jnp.zeros((*shape[:-1],self.determinants*self.spins[1]), dtype=dtype)
                    ][kernel_scale], axis=-1)
                )
            else:
                dense = nn.Dense(n_param, kernel_init=kernel_init(kernel_scale))
            # Actual orbital function
            return (dense(h) * IsotropicEnvelope(self.charges, out_dim, self.determinants, sigma_init=sigma_init, pi_init=pi_init)(r))\
                .reshape(-1, out_dim, self.determinants)

        # Case destinction for weight sharing 
        if self.share_weights:
            uu, dd = jnp.split(orbital_fn(h_one, r_im, max(self.spins)), self.spins[:1], axis=0)
            if self.full_det:
                ud, du = jnp.split(orbital_fn(
                    h_one,
                    r_im,
                    max(self.spins),
                    kernel_scale=0
                ), self.spins[:1], axis=0)
                orbitals = (jnp.concatenate([
                    jnp.concatenate([uu[:, :self.spins[0]], ud[:, :self.spins[1]]], axis=1),
                    jnp.concatenate([du[:, :self.spins[0]], dd[:, :self.spins[1]]], axis=1),
                ], axis=0),)
            else:
                orbitals = (uu[:, :self.spins[0]], dd[:, :self.spins[1]])
        else:
            h_by_spin = jnp.split(h_one, self.spins[:1], axis=0)
            r_im_by_spin = jnp.split(r_im, self.spins[:1], axis=0)
            orbitals = tuple(
                orbital_fn(h, r, d, kernel_scale=i)
                for h, r, d, i in zip(
                    h_by_spin,
                    r_im_by_spin,
                    (sum(self.spins),)*2 if self.full_det else self.spins,
                    (slice(None), slice(None, None, -1)) if self.full_det else (1.0,)*2
                )
            )
            if self.full_det:
                orbitals = (jnp.concatenate(orbitals, axis=0),)
        return tuple(o.transpose(2, 0, 1) for o in orbitals)


class LogSumDet(nn.Module):
    @nn.compact
    def __call__(self, xs):
        # Special case for 1x1 matrices
        # Here we avoid going into the log domain
        det1 = functools.reduce(
            lambda a, b: a*b,
            [x.reshape(-1) for x in xs if x.shape[-1] == 1],
            jnp.ones(())
        )

        sign_in, logdet = functools.reduce(
            lambda a, b: (a[0]*b[0], a[1]+b[1]),
            [jnp.linalg.slogdet(x) for x in xs if x.shape[-1] > 1],
            (jnp.ones(()), jnp.zeros(()))
        )

        maxlogdet = jax.lax.stop_gradient(jnp.max(logdet))
        det = sign_in * det1 * jnp.exp(logdet - maxlogdet)

        w = self.param(
            'w',
            nn.initializers.ones,
            det.shape
        )
        result = jnp.vdot(w, det)

        sign_out = jnp.sign(result)
        log_out = jnp.log(jnp.abs(result)) + maxlogdet
        return sign_out, log_out


class Jastrow(nn.Module):
    spins: tuple[int, int]
    
    @nn.compact
    def __call__(self, r_ij):
        a_par_w, a_anti_w = self.param(
            'weight',
            nn.initializers.constant(1e-2),
            (2,)
        )
        a_par, a_anti = self.param(
            'alpha',
            nn.initializers.ones,
            (2,)
        )
        r_ij = r_ij[..., -1]
        uu, ud, du, dd = [
            s
            for split in jnp.split(r_ij, self.spins[:1], axis=0)
            for s in jnp.split(split, self.spins[:1], axis=1)
        ]
        same = jnp.concatenate([uu.reshape(-1), dd.reshape(-1)])
        diff = jnp.concatenate([ud.reshape(-1), du.reshape(-1)])
        result = -(1/4) * a_par_w * (a_par**2 / (a_par + same)).sum()
        result += -(1/2) * a_anti_w * (a_anti**2 / (a_anti + diff)).sum()
        return result



class FermiNet(nn.Module):
    charges: Tuple[int, ...]
    spins: Tuple[int, int]
    hidden_dims: Sequence[Tuple[int, int]] = (
        (256, 32), (256, 32), (256, 32), (256, 32))
    determinants: int = 16
    full_det: bool = False
    input_config: dict = field(default_factory=lambda *_: {
        'activation': 'tanh',
        'nuclei_embedding': 64,
        'out_dim': 64,
        'mlp_depth': 2
    })
    jastrow_config: Optional[dict] = field(default_factory=lambda *_: {
        'activation': 'silu',
        'n_layers': 3
    })
    activation: str = "silu"
    absolute_spins: bool = False
    update_pair_independent: bool = False

    def setup(self):
        self.axes = self.variable(
            'constants',
            'axes',
            jnp.eye,
            3
        )
        self.input_construction = Embedding(
            **self.input_config
        )
        # Do not compute an update for the last pairwise layer
        hidden_dims = [list(h) for h in self.hidden_dims]
        hidden_dims[-1][1] = 0
        self.fermi_layers = [
            FermiLayer(
                n_elec=self.spins,
                single_size=single,
                pair_size=pair,
                activation=self.activation,
                absolute_spin=self.absolute_spins,
                update_pair_independent=self.update_pair_independent
            )
            for single, pair in hidden_dims
        ]

        self.to_orbitals = Orbitals(self.spins, self.charges, self.determinants, self.full_det, not self.absolute_spins)

        self.logsumdet = LogSumDet()
        if self.jastrow_config is not None:
            self.jastrow = AutoMLP(1, **self.jastrow_config)
            self.jastrow_weight = self.param(
                'jastrow_weight',
                nn.initializers.zeros,
                ()
            )
        self.cusp_jastorw = Jastrow(self.spins)

    def encode(self, electrons, atoms):
        # Prepare input
        atoms = atoms.reshape(-1, 3) @ self.axes.value
        electrons = electrons.reshape(-1, 3) @ self.axes.value
        h_one, h_two, r_im, r_ij = self.input_construction(electrons, atoms)

        # Fermi interaction
        for fermi_layer in self.fermi_layers:
            h_one, h_two = fermi_layer(h_one, h_two)

        return h_one, r_im, r_ij

    def orbitals(self, electrons, atoms):
        h_one, r_im, _ = self.encode(electrons, atoms)
        return self.to_orbitals(h_one, r_im)

    def signed(self, electrons, atoms):
        # Compute orbitals
        h_one, r_im, r_ij = self.encode(electrons, atoms)
        orbitals = self.to_orbitals(h_one, r_im)
        # Compute log det
        sign, log_psi = self.logsumdet(orbitals)

        # Optional jastrow factor
        if self.jastrow_config is not None:
            log_psi += self.jastrow(h_one).mean() * self.jastrow_weight
        log_psi += self.cusp_jastorw(r_ij)

        return sign, log_psi

    def __call__(self, electrons, atoms):
        return self.signed(electrons, atoms)[1]