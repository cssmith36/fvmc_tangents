import os

import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp

import fvmc
from fvmc.observable import project_spins
from fvmc.wavefunction.hegspin import (ElecProductModelSpin,
                                       PairJastrowCCKSpin, PlanewaveSlaterSpin,
                                       det_row_update, multi_det_row_update)

_key0 = jax.random.PRNGKey(0)

_cell3d = jnp.eye(3) * 10.
_cell2d = jnp.array([[6., 0.], [2., 6.]])

_nelec = 16


def _check_pbc_spin(key, model, params, cell):
    # check pbc
    key1, key2, key3 = jax.random.split(key, num=3)
    x = jax.random.uniform(key1, (_nelec, cell.shape[-1]))
    s = jax.random.uniform(key3, (1,_nelec))[0]*2.0*np.pi
    shift = jax.random.randint(key2, (_nelec, cell.shape[0],), -3, 3)
    x_shift = x + shift @ cell
    out1 = model.apply(params, (x,s))
    out2 = model.apply(params, (x_shift,s))
    chex.assert_trees_all_close(out2, out1)


def _check_perm_spin(key, model, params, cell, anti_symm=True):
    # check permutation
    key1, key2 = jax.random.split(key, num=2)
    x = jax.random.uniform(key1, (_nelec, cell.shape[-1]))
    s = jax.random.uniform(key2, (1,_nelec))[0]*2.0*np.pi
    perm = jnp.arange(_nelec, dtype=int).at[:2].set([1,0])
    x_perm = x[perm]
    s_perm = s[perm]
    sign1, logf1 = model.apply(params, (x,s))
    sign2, logf2 = model.apply(params, (x_perm, s_perm))
    assert not jnp.iscomplexobj(logf1)
    np.testing.assert_allclose(sign2, -sign1 if anti_symm else sign1)
    np.testing.assert_allclose(logf2, logf1)


@pytest.mark.parametrize("dtype", [float, complex])
def test_det_row_updates(dtype):
    n = 6
    key1, key2 = jax.random.split(_key0)
    mat1 = jax.random.normal(key1, (n, n), dtype=dtype)
    mat2 = jax.random.normal(key2, (n, n), dtype=dtype)
    det1 = jnp.linalg.det(mat1)
    new_dets = jnp.stack([
        jnp.linalg.det(mat1.at[ii].set(mat2[ii])) for ii in range(n)
    ])
    row_diffs = mat2 - mat1
    updates = det_row_update(mat1, row_diffs)
    np.testing.assert_allclose(updates, new_dets / det1)


@pytest.mark.parametrize("dtype", [float, complex])
def test_multi_det_row_updates(dtype):
    m = 3
    n = 6
    key1, key2, key3 = jax.random.split(_key0, 3)
    mat1 = jax.random.normal(key1, (m, n, n), dtype=dtype)
    mat2 = jax.random.normal(key2, (m, n, n), dtype=dtype)
    weights = jax.random.uniform(key3, (m,))
    det1 = jnp.linalg.det(mat1) @ weights
    new_dets = jnp.stack([
        jnp.linalg.det(mat1.at[:, ii].set(mat2[:, ii])) for ii in range(n)
    ], axis=0) @ weights
    row_diffs = mat2 - mat1
    updates = multi_det_row_update(mat1, row_diffs, weights)
    np.testing.assert_allclose(updates, new_dets / det1)


@pytest.mark.parametrize("multi_det", [None, 4])
@pytest.mark.parametrize("cell", [_cell3d, _cell2d])
def test_planewave_slater_spin(cell, multi_det):
    model = PlanewaveSlaterSpin(cell=cell, multi_det=multi_det)
    params = model.init(_key0, (jnp.zeros((_nelec, cell.shape[-1])), jnp.zeros(_nelec)))
    _check_pbc_spin(_key0, model, params, cell)
    _check_perm_spin(_key0, model, params, cell, anti_symm=True)


@pytest.mark.parametrize("cell", [_cell3d, _cell2d])
@pytest.mark.parametrize("optimize_cusp", [True, False])
def test_pair_jastrow_cck(cell, optimize_cusp):
    model = PairJastrowCCKSpin(cell=cell, optimize_cusp=optimize_cusp)
    params = model.init(_key0, (jnp.zeros((_nelec, cell.shape[-1])), jnp.zeros(_nelec)))
    _check_pbc_spin(_key0, model, params, cell)
    _check_perm_spin(_key0, model, params, cell, anti_symm=False)


@pytest.mark.parametrize("cell", [_cell3d, _cell2d])
@pytest.mark.parametrize("multi_det", [4, None])
@pytest.mark.parametrize("with_jastrow", [True, False])
def test_apply_pauli(cell, multi_det, with_jastrow):
    model = PlanewaveSlaterSpin(cell=cell, multi_det=multi_det)
    if with_jastrow:
        model = ElecProductModelSpin([model, PairJastrowCCKSpin(cell=cell)])
    params = model.init(_key0, (jnp.zeros((_nelec, cell.shape[-1])), jnp.zeros(_nelec)))
    key1, key2 = jax.random.split(_key0)
    x = jax.random.uniform(key1, (_nelec, cell.shape[-1]))
    s = jax.random.uniform(key2, (1,_nelec))[0]*2.0*np.pi
    pauli = model.apply(params, (x, s), method='pauli')
    ref = project_spins(model, params, (x, s))
    np.testing.assert_allclose(pauli, ref, atol=1e-6)


def _common_config(n_elec=12, n_k=36, seed=42, iterations=10000):
    rs = 30
    sig = 2.6 * rs**(3/4)
    cell = np.diag([1.714163052355123114e+02, 1.979344999424281752e+02])
    wgrid = np.mgrid[:3, :4].transpose(1, 2, 0).reshape(-1, 2)
    wcpos = wgrid * np.array([1/3, 1/4]) @ cell

    # ansatz
    ansatz = PlanewaveSlaterSpin(
        cell=cell, # shape (ndim, ndim)
        n_k=n_k, # number of k-points
        # below are all default values, written here for demonstration
        multi_det=None, # None or interger representing the number of determinants
        twist=None, # single twist, shape (ndim,), between 0 and 1
        close_shell=False, # whether to round up k points to closed-shell
        init_scale=0.01, # the scale of initial random parameters
    )
    # initialize random parameters (do some weird stuff to make orbitals more distinct)
    key = jax.random.PRNGKey(seed)
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    #define pseudo spin up number to make orbitals more distinct for different random seeds
    n_up = jax.random.randint(subkey1, shape=(1,), minval=1, maxval=n_elec-1)[0]
    n_up = 12
    n_dn = n_elec - n_up
    print('n_up,n_dn',n_up,n_dn)
    # orbital coeff for spin up basis (random & fixed part)
    coeffup = jax.random.uniform(key, shape=(n_k, n_elec), minval=-1.0, maxval=1.0)
    # assign to params
    params = ansatz.init(key, (jnp.zeros((n_elec, 2)), jnp.zeros(n_elec)))
    print('coeeff shape', coeffup.shape)
    params['params']['orbital_fn']['VmapDense_0']['kernel'] = coeffup

    # init coords and spins for sampler
    def conf_init_fn(key):
        key1, key2 = jax.random.split(key)
        coords = wcpos + 0.1 * sig * jax.random.normal(key1, wcpos.shape)
        spins = jax.random.uniform(key2, (n_elec,)) * 2 * jnp.pi
        return coords, spins

    # build config for fvmc run
    cfg = fvmc.config.default()
    cfg.seed = seed
    cfg.verbosity = "INFO"
    # <system>
    cfg.system.charge = -n_elec
    cfg.system.elems = None
    cfg.system.nuclei = None
    cfg.system.cell = cell
    # <sample>
    cfg.sample.size = 10240
    cfg.sample.burn_in = 100
    cfg.sample.sampler = 'hmc'
    cfg.sample.hmc.dt = 0.025 * rs
    cfg.sample.hmc.length = None
    cfg.sample.hmc.steps = 50
    cfg.sample.hmc.grad_clipping = 2.
    cfg.sample.hmc.speed_limit = 5.
    # cfg.sample.hmc.jitter_dt = 0.2
    cfg.sample.hmc.segments = 5
    cfg.sample.hmc.div_threshold = 5
    cfg.sample.hmc.mass = (1., 500.)
    cfg.sample.conf_init_fn = conf_init_fn
    # <energy>
    cfg.loss.energy_clipping = None #5.
    cfg.loss.clip_from_median = True
    # <optimize>
    cfg.optimize.iterations = iterations
    cfg.optimize.optimizer = 'adabelief'
    cfg.optimize.lr.base = 1e-3 # 1e-2
    cfg.optimize.lr.decay_time = 100
    # <output>
    cfg.log.stat_every = 1 #10
    cfg.log.dump_every = 100
    cfg.log.ckpt_keep = 1 # None
    cfg.log.ckpt_every = 1000
    cfg.log.use_tensorboard = False
    # <ansatz>
    cfg.ansatz = ansatz
    cfg.restart.params = params

    return cfg

def _common_spin_config(n_elec=12, n_k=36, seed=42, iterations=100):
    rs = 30
    sig = 2.6 * rs**(3/4)
    cell = np.diag([1.714163052355123114e+02, 1.979344999424281752e+02])
    wgrid = np.mgrid[:3, :4].transpose(1, 2, 0).reshape(-1, 2)
    wcpos = wgrid * np.array([1/3, 1/4]) @ cell

    # ansatz
    ansatz = PlanewaveSlaterSpin(
        cell=cell, # shape (ndim, ndim)
        n_k=n_k, # number of k-points
        # below are all default values, written here for demonstration
        multi_det=None, # None or interger representing the number of determinants
        twist=None, # single twist, shape (ndim,), between 0 and 1
        close_shell=False, # whether to round up k points to closed-shell
        init_scale=0.01, # the scale of initial random parameters
    )
    # initialize random parameters (do some weird stuff to make orbitals more distinct)
    key = jax.random.PRNGKey(seed)
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    #define pseudo spin up number to make orbitals more distinct for different random seeds
    n_up = jax.random.randint(subkey1, shape=(1,), minval=1, maxval=n_elec-1)[0]
    #n_up = 12
    n_dn = n_elec - n_up
    print('n_up,n_dn',n_up,n_dn)
    # orbital coeff for spin up basis (random & fixed part)
    coeffup = jax.random.uniform(key, shape=(n_k, n_elec), minval=-1.0, maxval=1.0)
    block1 = jnp.eye(n_up)
    block2 = jnp.zeros((n_up, n_dn))
    block3 = jnp.zeros((n_k - n_up, n_elec))
    coeff1 = jnp.vstack((jnp.hstack((block1, block2)), block3))
    # orbital coeff for spin dn basis (random & fixed part)
    coeffdn = jax.random.uniform(subkey3, shape=(n_k, n_elec), minval=-1.0, maxval=1.0)
    block1 = jnp.zeros((n_dn, n_up))
    block2 = jnp.eye(n_dn)
    block3 = jnp.zeros((n_k - n_dn, n_elec))
    coeff2 = jnp.vstack((jnp.hstack((block1, block2)), block3))
    # randomize and mix
    factor_random = jax.random.uniform(subkey2, shape=(5,), minval=-8.0, maxval=8.0)
    coeffup = coeff1 * factor_random[0] + coeffup * factor_random[4] / 4.0
    coeffdn = coeff2 * factor_random[2] + coeffdn * factor_random[3]
    coeff = jnp.stack((coeffup, coeffdn), axis=-1)
    # assign to params
    params = ansatz.init(key, (jnp.zeros((n_elec, 2)), jnp.zeros(n_elec)))
    params['params']['orbital_fn']['VmapDense_0']['kernel'] = coeff

    # init coords and spins for sampler
    def conf_init_fn(key):
        key1, key2 = jax.random.split(key)
        coords = wcpos + 0.1 * sig * jax.random.normal(key1, wcpos.shape)
        spins = jax.random.uniform(key2, (n_elec,)) * 2 * jnp.pi
        return coords, spins

    # build config for fvmc run
    cfg = fvmc.config.default()
    cfg.seed = seed
    cfg.verbosity = "INFO"
    # <system>
    cfg.system.charge = -n_elec
    cfg.system.elems = None
    cfg.system.nuclei = None
    cfg.system.cell = cell
    # <sample>
    cfg.sample.size = 10240
    cfg.sample.burn_in = 100
    cfg.sample.sampler = 'hmc'
    cfg.sample.hmc.dt = 0.025 * rs
    cfg.sample.hmc.length = None
    cfg.sample.hmc.steps = 50
    cfg.sample.hmc.grad_clipping = 2.
    cfg.sample.hmc.speed_limit = 5.
    # cfg.sample.hmc.jitter_dt = 0.2
    cfg.sample.hmc.segments = 5
    cfg.sample.hmc.div_threshold = 5
    cfg.sample.hmc.mass = (1., 500.)
    cfg.sample.conf_init_fn = conf_init_fn
    # <energy>
    cfg.loss.energy_clipping = None #5.
    cfg.loss.clip_from_median = True
    # <optimize>
    cfg.optimize.iterations = iterations
    cfg.optimize.optimizer = 'sgd'
    cfg.optimize.lr.base = 0 # 1e-2
    cfg.optimize.lr.decay_time = 100
    # <output>
    cfg.log.stat_every = 1 #10
    cfg.log.dump_every = 100
    cfg.log.ckpt_keep = 1 # None
    cfg.log.ckpt_every = 1000
    cfg.log.use_tensorboard = False
    # <ansatz>
    cfg.ansatz = ansatz
    cfg.restart.params = params

    return cfg


@pytest.mark.veryslow
def test_slater_spin_wick_eval(tmp_path):
    from fvmc.ewaldsum import gen_lattice_displacements, gen_pbc_disp_fn
    from fvmc.utils import displace_matrix, gen_kidx, split_spin

    # work in temporary directory
    os.chdir(tmp_path)

    # system constants
    n_elec = 12
    n_k = 3 * n_elec
    seed = 42
    iterations = 100
    cfg = _common_spin_config(n_elec=n_elec, n_k=n_k,
                              seed=seed, iterations=iterations)

    # gaussian potential
    lamb = 0.08 # 0.04 #0.001 #0.2
    pref = 0.5 #1e2
    n_lat = 1
    cell = latvec = jnp.asarray(cfg.system.cell)
    recvec = jnp.linalg.inv(latvec).T
    kpts = jnp.asarray(gen_kidx(2, n_k, close_shell=False)) #n_dim = 2
    klist = 2 * jnp.pi * kpts @ recvec # [n_k, n_dim]

    cellvolume = jnp.abs(jnp.linalg.det(latvec))
    disp_fn = gen_pbc_disp_fn(latvec)
    lat_disp = gen_lattice_displacements(latvec, n_lat)
    lat_norm = jnp.linalg.norm(lat_disp, axis=-1)
    lat_norm = lat_norm[lat_norm > 0]
    self_img = pref * jnp.sum(jnp.exp(-0.5* lamb * lat_norm * lat_norm))

    def gaussian_x(charge, pos): #AV MOVE TO TEST
        if charge.shape[0] < 2:
            return 0
        disp = displace_matrix(pos, pos, disp_fn=disp_fn)
        rvec = disp[None, :, :, :] + lat_disp[:, None, None, :]
        r = jnp.linalg.norm(rvec + jnp.eye(pos.shape[0])[..., None], axis=-1)
        charge_ij = charge[:, None] * charge[None, :]
        e_real = pref * jnp.sum(jnp.triu(charge_ij * jnp.exp(-0.5 * lamb * r * r), k=1))
        e_real += 0.5 * jnp.sum(charge ** 2) * self_img # self image
        e_bg = jnp.sum(charge**2)**2*pref*2.0*jnp.pi/lamb*0.5/cellvolume
        e_real = e_real - e_bg
        return e_real

    def calc_pe(elems, r, x):
        """Warpped interface for potential energy from nuclei and electrons"""
        assert elems.shape[0] == r.shape[0]
        x, _ = split_spin(x) # Coulomb is spin independent
        assert elems.ndim == 1 and r.ndim == x.ndim == 2
        charge = jnp.concatenate([elems, -jnp.ones(x.shape[0])], axis=0)
        pos = jnp.concatenate([r, x], axis=0)
        return gaussian_x(charge, pos)

    cfg.loss.pe_kwargs = lambda *a, **k: calc_pe(*a)

    ############## SLOW BELOW ###############
    _ = fvmc.train.main(cfg) # run fvmc
    ############## SLOW ABOVE ###############

    # collect results
    data = np.loadtxt("data.txt")
    n_iter = data.shape[0]
    with open("data.txt", "r") as f:
        header = f.readline().strip("#").split()
    iek = header.index("e_kin")
    ek_mean = data[:, iek].mean()
    ek_std = data[:, iek].std() / np.sqrt(n_iter)
    iep = header.index("e_coul")
    ep_mean = data[:, iep].mean()
    ep_std = data[:, iep].std() / np.sqrt(n_iter)
    iet = header.index("e_tot")
    et_mean = data[:, iet].mean()
    et_std = data[:, iet].std() / np.sqrt(n_iter)
    print("from fvmc:")
    print("ekin:", ek_mean, "+-", ek_std)
    print("epot:", ep_mean, "+-", ep_std)
    print("etot:", et_mean, "+-", et_std)

    # calculate reference
    n_sigma=2
    lx = cell[0, 0]
    ly = cell[1, 1]
    area = lx * ly
    dkx= 2 * np.pi / lx
    dky= 2 * np.pi / ly
    n_qmax = int(np.max(np.abs(klist)) / min(dkx, dky)) * 2
    q1x = np.arange(-n_qmax, n_qmax + 1) * dkx
    q1y = np.arange(-n_qmax, n_qmax + 1) * dky
    qlist = np.stack(np.meshgrid(q1x, q1y, indexing='ij'), axis=-1).reshape(-1, 2)
    n_q = qlist.shape[0]
    # w is coeff, but pull the spin dim to the front
    coeff = cfg.restart.params['params']['orbital_fn']['VmapDense_0']['kernel']
    w = np.moveaxis(np.asarray(coeff), -1, 0) # [n_spin, n_k, n_orb(=n_elec)]
    n_orb = w.shape[2]
    _w = w.reshape(n_sigma * n_k, n_orb)
    _q, _r = np.linalg.qr(_w)
    w = _q.reshape(n_sigma, n_k, n_orb)

    def gaussian_q(qvec):
        qnorm = np.linalg.norm(qvec, axis=-1) #+(1e-8)**2)
        v = 2 * np.pi / lamb * pref * np.exp(-qnorm**2 / (2 * lamb))
        return v

    def _check_in_arr(el, ar):
        """
        el : dim 2
        ar : dim Nx2
        """
        diff = np.linalg.norm(el - ar, axis=1)
        return np.any(np.isclose(0, diff))

    def _get_idx_in_arr(el, ar):
        """
        el : dim 2
        ar : dim Nx2
        """
        diff = np.linalg.norm(el - ar, axis=1)
        idx = np.argmin(diff)
        assert np.isclose(diff[idx], 0)
        return idx

    def get_w_plus(w):
        w_plus = np.zeros((n_sigma, n_k, n_q, n_orb), dtype=complex)
        for iq in range(n_q):
            for ik1 in range(n_k):
                    k1pq = klist[ik1] + qlist[iq]
                    if _check_in_arr(k1pq, klist):
                        ind1 = _get_idx_in_arr(k1pq, klist)
                        w_plus[:, ik1, iq, :] = w[:, ind1, :]
        return w_plus

    def get_w_minus(w):
        w_minus = np.zeros((n_sigma, n_k, n_q, n_orb), dtype=complex)
        for iq in range(n_q):
            for ik2 in range(n_k):
                    k2mq = klist[ik2] - qlist[iq]
                    if _check_in_arr(k2mq, klist):
                        ind2 = _get_idx_in_arr(k2mq, klist)
                        w_minus[:, ik2, iq, :]=w[:, ind2, :]
        return w_minus

    def calc_e_kin(w):
        ekin_pref = (klist[:, 0]**2 + klist[:, 1]**2) / 2
        ekin = np.einsum('ijk, j ->', w.conj() * w, ekin_pref)
        return ekin

    def calc_e_pot(w):
        w_plus = get_w_plus(w)
        w_minus = get_w_minus(w)
        v_pref = 1 / (2 * area) * gaussian_q(qlist)
        e_ha =  np.einsum('sija, sia, j, tljb, tlb',
                        w_plus.conj(), w, v_pref, w_minus.conj(), w).real
        e_fo = -np.einsum('sija, sib, j, tljb, tla', \
                        w_plus.conj(), w, v_pref, w_minus.conj(), w).real
        e_bg = -0.5 * gaussian_q(np.array([0,0])) * n_elec**2 / area
        return e_ha, e_fo, e_bg

    ref_epot = (np.sum(calc_e_pot(w)))
    ref_ekin = calc_e_kin(w)
    ref_etot = ref_ekin + ref_epot
    print("reference:")
    print("ekin:", ref_ekin)
    print("epot:", ref_epot)
    print("etot:", ref_etot)

    # compare
    assert ek_std < 3e-4 and ep_std < 3e-4 and et_std < 3e-4
    np.testing.assert_allclose(ek_mean, ref_ekin, atol=3 * ek_std)
    np.testing.assert_allclose(ep_mean, ref_epot, atol=3 * ep_std)
    np.testing.assert_allclose(et_mean, ref_etot, atol=3 * et_std)


@pytest.mark.veryslow
def test_slater_spin_wick_pauli(tmp_path):
    from fvmc.utils import gen_kidx

    # work in temporary directory
    os.chdir(tmp_path)

    # system constants
    n_elec = 12
    n_k = 3 * n_elec
    seed = 4242
    iterations = 100
    cfg = _common_spin_config(n_elec=n_elec, n_k=n_k,
                              seed=seed, iterations=iterations)

    # random spin potential
    v = jax.random.uniform(jax.random.PRNGKey(seed+1), shape=(1, 3),
                           minval=-1.0, maxval=1.0)
    def spinpotential(x):
        nx= x.shape[0]
        v_stacked = jnp.tile(v, (nx, 1))
        return v_stacked

    cfg.system.spin_potentials = {"rand": lambda x: spinpotential(x),
                                  "rand1b": lambda x: v.reshape(-1)}
    cfg.loss.ke_kwargs = lambda *a, **k: 0.
    cfg.loss.pe_kwargs = lambda *a, **k: 0.

    ############## SLOW BELOW ###############
    _ = fvmc.train.main(cfg) # run fvmc
    ############## SLOW ABOVE ###############

    # collect results
    data = np.loadtxt("data.txt")
    n_iter = data.shape[0]
    with open("data.txt", "r") as f:
        header = f.readline().strip("#").split()
    ie1 = header.index("e_rand")
    ie2 = header.index("e_rand1b")
    np.testing.assert_allclose(data[:, ie1], data[:, ie2], rtol=1e-12)
    espin_mean = data[:, ie1].mean()
    espin_std = data[:, ie1].std() / np.sqrt(n_iter)
    print("from fvmc:")
    print("espin:", espin_mean, "+-", espin_std)

    # calculate reference
    n_sigma=2
    # w is coeff, but pull the spin dim to the front
    coeff = cfg.restart.params['params']['orbital_fn']['VmapDense_0']['kernel']
    w = np.moveaxis(np.asarray(coeff), -1, 0) # [n_spin, n_k, n_orb(=n_elec)]
    n_orb = w.shape[2]
    _w = w.reshape(n_sigma * n_k, n_orb)
    _q, _r = np.linalg.qr(_w)
    w = _q.reshape(n_sigma, n_k, n_orb)

    tspin=np.zeros((n_k, n_sigma, n_sigma), dtype=complex)
    # sigma x
    tspin[:,0,1]+=1*v[0,0]
    tspin[:,1,0]+=1*v[0,0]
    # sigma y
    tspin[:,0,1]+=-1j*v[0,1]
    tspin[:,1,0]+=1j*v[0,1]
    # sigma z
    tspin[:,0,0]+=1*v[0,2]
    tspin[:,1,1]+=-1*v[0,2]
    # reference energy
    ref_espin = np.einsum('ijk,jis,kis->', tspin, w.conj(), w).real
    print("reference:")
    print("espin:", ref_espin)

    # compare
    assert espin_std < 5e-3
    np.testing.assert_allclose(espin_mean, ref_espin, atol=3 * espin_std)


@pytest.mark.veryslow
def test_slater_spin_pauli_optimize(tmp_path):
    # work in temporary directory
    os.chdir(tmp_path)

    # system constants
    n_elec = 12
    n_k = 3 * n_elec
    seed = 4242
    iterations = 200
    cfg = _common_spin_config(n_elec=n_elec, n_k=n_k,
                              seed=seed, iterations=iterations)
    cfg.optimize.lr.base = 1
    cfg.optimize.optimizer = 'adabelief'

    # sigma z spin potential
    v = jnp.array([0.0, 0.0, 1.0])
    cfg.system.spin_potentials = {"sz1b": lambda x: v}
    cfg.loss.ke_kwargs = lambda *a, **k: 0.
    cfg.loss.pe_kwargs = lambda *a, **k: 0.

    ############## SLOW BELOW ###############
    _ = fvmc.train.main(cfg) # run fvmc
    ############## SLOW ABOVE ###############

    # collect results
    data = np.loadtxt("data.txt")
    nsample = cfg.sample.size
    with open("data.txt", "r") as f:
        header = f.readline().strip("#").split()
    last_mean = data[-1, header.index("e_tot")]
    last_std = data[-1, header.index("std_e")]
    last_err = last_std / np.sqrt(nsample)
    print("final results:", last_mean, "+-", last_err)
    assert last_std < 1e-4
    np.testing.assert_allclose(last_mean, -n_elec, atol=3 * last_err)

#test_slater_spin_wick_eval('/mnt/home/csmith1/ceph/excitedStates/fvmc/tests/wavefunction/tmp')