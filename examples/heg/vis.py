#!/usr/bin/env python3
import yaml
import numpy as np
import matplotlib.pyplot as plt
from fvmc import observable as obs
from fvmc.wavefunction.heg import heg_rs

def bin_centers(edges):
    ndim = len(edges)
    mids = [0.5*(e[1:]+e[:-1]) for e in edges]
    centers = np.stack(
        np.meshgrid(*mids, indexing='ij')
    , axis=-1).reshape(-1, ndim)
    return centers

def spin_split(spins, yms, yes):
    nspin = len(spins)
    if nspin == 1:  # no spin to split
      return yms, yes
    npair = len(yms)
    # !!!! hard-code for up dn
    if npair == (nspin+1)*nspin//2:
      same_diff_mean = np.array([yms[0]+yms[2], 2*yms[1]])
      same_diff_error = np.array([(yes[0]**2+yes[2]**2)**0.5, 2*yes[1]])
    elif npair == nspin:
      same_diff_mean = np.array([yms[0]+yms[1], yms[0]-yms[1]])
      same_diff_error = np.array([(yes[0]**2+yes[1]**2)**0.5,]*2)
    else:
      msg = 'nspin = %d; npair = %d' % (nspin, npair)
      raise RuntimeError(msg)
    return same_diff_mean, same_diff_error

def spin_sum(same_diff_mean, same_diff_error):
    ym = same_diff_mean.sum(axis=0)
    ye = (same_diff_error**2).sum(axis=0)**0.5
    return ym, ye

def show_rhos(fig, rvecs, rhos, colorbar=False):
    nspin = len(rhos)
    mesh = rhos.shape[1:]
    nnr = np.prod(mesh)
    rho_tot = rhos.sum(axis=0)
    #nelec = int(round(rho_tot.sum()/nnr))
    #title_tot = 'charge density N=%d' % nelec
    if nspin == 1:
      ax = fig.add_subplot(1, 1, 1, aspect=1)
      #ax.set_title(title_tot)
      axl = [ax]
      cs = contour_scatter(ax, rvecs, rho_tot.ravel(), mesh=mesh)
      if colorbar: plt.colorbar(cs)
    elif nspin == 2:
      axl = []
      for i in range(2):
        ax = fig.add_subplot(1, 2, i+1, aspect=1)
        axl.append(ax)

      ax = axl[0]
      #ax.set_title(title_tot)
      cs = contour_scatter(ax, rvecs, rho_tot.ravel(), mesh=mesh)
      if colorbar: plt.colorbar(cs)

      ax = axl[1]
      #ax.set_title('spin density up-dn')
      drho = rhos[0]-rhos[1]
      smax = max(drho.max(), abs(drho.min()))
      zlim = (-smax, smax)
      cs = contour_scatter(ax, rvecs, drho.ravel(), mesh=mesh, zlim=zlim, cmap='coolwarm')
      if colorbar: plt.colorbar(cs)
    else:
      msg = 'nspin = %d in show_rhos' % nspin
      raise RuntimeError(msg)
    return axl

def show_mags(fig, rvecs, mags, rhom=None, colorbar=False):
    mesh = mags.shape[-2:]
    axl = []
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, aspect=1)
        axl.append(ax)
    axl[0].set_title('total')
    if rhom is not None:
        cs = contour_scatter(axl[0], rvecs, rhom.ravel(), mesh=mesh)
        if colorbar:
            plt.colorbar(cs)
    names = ['x', 'y', 'z']
    for ax, mag, name in zip(axl[1:], mags, names):
        ax.set_title(name)
        cs = contour_scatter(ax, rvecs, mag.ravel(), mesh=mesh)
        if colorbar:
            plt.colorbar(cs)
    return axl

# from qharv with MIT (qharv.field.kyrt.set_style)
def set_style(style='ticks', context='talk', **kwargs):
  import seaborn as sns
  if (context=='talk') and ('font_scale' not in kwargs):
    kwargs['font_scale'] = 0.9
  sns.set_style(style)
  sns.set_context(context, **kwargs)

# from qharv with MIT (qharv.field.kyrt.contour_scatter)
def contour_scatter(ax, xy, z, zlim=None, nz=8, cmap='viridis', fill=True,
  interp_method='linear', mesh=(32, 32), lims=None, **kwargs):
  """View sampled scalar field using contours

  Args:
    ax (plt.Axes): matplotlib axes
    xy (np.array): scatter points, a list of 2D vectors
    z (np.array): scatter values, one at each scatter point
    zlim (list, optional): value (min, max) for colormap
    cmap (str, optional): color map name, default is 'viridis'
    nz (int, optional): number of contour lines for when zlim is set
    interp_method (str, optional): griddata, default 'linear'
    mesh (tuple, optional): regular grid shape, default (32, 32)
    kwargs (dict, optional): keyword arguments to be passed to ax.scatter
  Returns:
    matplotlib.contour.QuadContourSet: filled contour plot
  Example:
    >>> kxy = kvecs[:, :2]
    >>> nkxy = nofk[:, :, 0]
    >>> cs = contour_scatter(ax, kxy, nkxy, zlim=(0, 2), mesh=(256, 256))
  """
  import numpy as np
  from scipy.interpolate import griddata
  # interpret inputs and set defaults
  if zlim is not None:
    levels = np.linspace(*zlim, nz)
    if 'levels' in kwargs:
      msg = 'multiple values for keyward argument \'levels\''
      raise TypeError(msg)
    kwargs['levels'] = levels
  if lims is None:
    xarr = xy[:, 0]
    xmin = xarr.min()
    xmax = xarr.max()
    yarr = xy[:, 1]
    ymin = yarr.min()
    ymax = yarr.max()
    lims = ((xmin, xmax), (ymin, ymax))
  # create regular grid
  finex = np.linspace(lims[0][0], lims[0][1], mesh[0])
  finey = np.linspace(lims[1][0], lims[1][1], mesh[1])
  fine_points = [[(x, y) for y in finey] for x in finex]
  # interpolate scatter on regular grid
  interp_data = griddata(xy, z, fine_points, method=interp_method)
  finez = interp_data.reshape(*mesh).T
  # make contour plot
  if fill:
    cs = ax.contourf(finex, finey, finez, cmap=cmap, **kwargs)
  else:
    cs = ax.contour(finex, finey, finez, cmap=cmap, **kwargs)
  return cs

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('prefix', type=str)
  parser.add_argument('--spin_sum', '-ss', action='store_true')
  parser.add_argument('--savefig', '-s', action='store_true')
  parser.add_argument('--colorbar', '-cb', action='store_true')
  parser.add_argument('--verbose', '-v', action='store_true')
  args = parser.parse_args()
  prefix = args.prefix
  if prefix.endswith('-'):
    prefix = prefix[:-1]  # !!!! HACK: allow common tap completion typo

  meta, ym, ye = obs.load_obs(prefix)
  spins = np.asarray(meta['spins'])
  cell = np.asarray(meta['cell'])
  nspin = len(spins)
  nelec = spins.sum()
  rs = heg_rs(cell, nelec)

  set_style()
  obs_type = meta['aname']
  if obs_type == 'density':
    # interpret
    rhoms, rhoes = ym/nelec, ye/nelec
    edges = np.asarray([meta['edge0'], meta['edge1']])
    mesh = [len(e)-1 for e in edges]; nnr = np.prod(mesh)
    rvecs = bin_centers(edges)@cell/rs
    if args.spin_sum:
      nspin = 1
      rhoms = rhoms.sum(axis=0)
      rhoms = rhoms.reshape(1, *rhoms.shape)
    # visualize
    ny = 4
    fig = plt.figure(figsize=(nspin*ny+1, ny))
    axl = show_rhos(fig, rvecs, rhoms, args.colorbar)
    for ax in axl:
      ax.set_xlabel(r'$x/r_s$')
      ax.set_ylabel(r'$y/r_s$')
  elif obs_type == 'mdens':  # magnetization density
    try:
      prefix_dens = prefix.replace('mdens', 'dens')
      meta_rho, rhoms, rhoes = obs.load_obs(prefix_dens)
      rhom = rhoms.sum(axis=0)/nelec
    except:
      rhom = None
    # interpret
    magm, mage = ym[0], ye[0]
    edges = np.asarray([meta['edge0'], meta['edge1']])
    mesh = [len(e)-1 for e in edges]; nnr = np.prod(mesh)
    rvecs = bin_centers(edges)@cell/rs
    # visualize
    fig = plt.figure(figsize=(12, 12))
    axl = show_mags(fig, rvecs, magm, rhom=rhom, colorbar=args.colorbar)
    for ax in axl:
      ax.set_xlabel(r'$x/r_s$')
      ax.set_ylabel(r'$y/r_s$')
  elif obs_type == 'gofr':
    if nspin == 2:
      # undo spin-dependent normalization
      n1, n2 = spins
      ym[0] = ym[0] * n1 * (n1-1) / 2
      ye[0] = ye[0] * n1 * (n1-1) / 2
      ym[2] = ym[2] * n2 * (n2-1) / 2
      ye[2] = ye[2] * n2 * (n2-1) / 2
      ym[1] = ym[1] * n1 * n2 / 2
      ye[1] = ye[1] * n1 * n2 / 2
      norm = nelec * (nelec-1) / 2
      ym = ym / norm
      ye = ye / norm
    # interpret
    grms, gres = spin_split(spins, ym, ye)
    if args.spin_sum:
      nspin = 1
      grm, gre = spin_sum(grms, gres)
      grms = grm[None]
      gres = gre[None]
    else:
      grms = np.array([grms[0]+grms[1], grms[0]-grms[1]])
      gres = np.array([(gres[0]**2+gres[1]**2)**0.5]*2)
    r = np.asarray(meta['r'])
    x = r/rs
    # visualize
    ylabel = 'g(r)'
    fig, axl = plt.subplots(nspin, 1, sharex=True)
    if nspin == 1:
      axl = [axl]
    else:
      axl[-1].set_ylabel('spin ' + ylabel)
      axl[-1].axhline(0, c='k', lw=0.6)
    axl[-1].set_xlabel(r'$r/r_s$')
    axl[0].set_ylabel(ylabel)
    axl[0].axhline(1, c='k', lw=0.6)
    for ax, grm, gre in zip(axl, grms, gres):
      ax.set_xlim(0, x.max())
      ax.errorbar(x, grm, gre)
  elif obs_type == 'vecgofr':
    # interpret
    gvms, gves = spin_split(spins, ym, ye)
    gvm, gve = spin_sum(gvms, gves)  # total
    edges = np.asarray([meta['edge0'], meta['edge1']])
    mesh = [len(e)-1 for e in edges]; nnr = np.prod(mesh)
    rvecs = bin_centers(edges)@cell/rs
    # visualize
    ny = 4
    fig = plt.figure(figsize=(nspin*ny+1, ny))
    axl = []
    for i in range(nspin):
      ax = fig.add_subplot(1, nspin, i+1, aspect=1)
      axl.append(ax)
    for ax in axl:
      ax.set_xlabel(r'$x/r_s$')
      ax.set_ylabel(r'$y/r_s$')
    #   total
    ax = axl[0]
    cs = contour_scatter(ax, rvecs, gvm.ravel(), mesh=mesh)
    if args.colorbar: plt.colorbar(cs, fraction=0.046, pad=0.04)
    #   difference
    if nspin > 1:
      ax = axl[1]
      dgvm = gvms[0]-gvms[1]  # difference
      dgve = (gves[0]**2+gves[1]**2)**0.5
      smax = max(dgvm.max(), abs(dgvm.min()))
      zlim = (-smax, smax)
      cs = contour_scatter(ax, rvecs, dgvm.ravel(), mesh=mesh, zlim=zlim, cmap='coolwarm')
      if args.colorbar: plt.colorbar(cs)
  elif obs_type in ['sofk', 'dsk', 'rhok']:
    # interpret
    skms, skes = spin_split(spins, ym, ye)
    skms /= nelec
    skes /= nelec
    if args.spin_sum:
      nspin = 1
      skm, ske = spin_sum(skms, skes)
      skms = skm[None]
      skes = ske[None]
    else:
      skms = np.array([skms[0]+skms[1], skms[0]-skms[1]])
      skes = np.array([(skes[0]**2+skes[1]**2)**0.5]*2)
    kvecs = np.asarray(meta['kvecs'])
    kmags = np.linalg.norm(kvecs, axis=-1)
    x = kmags*rs
    # visualize
    ylabel = {'sofk': 'S(k)', 'dsk': r'$\delta$S(k)', 'rhok': r'$Re[\rho_k]$'}[obs_type]
    fig, axl = plt.subplots(nspin, 1, sharex=True)
    if nspin == 1:
      axl = [axl]
    else:
      axl[-1].set_ylabel('spin ' + ylabel)
    axl[-1].set_xlabel(r'$k r_s$')
    ax = axl[0]
    ax.set_ylabel(ylabel)
    for ax, skm, ske in zip(axl, skms, skes):
      ax.set_xlim(0, x.max())
      ymax = skm.real.max()
      ax.set_ylim(0, 1.05*ymax)
      ax.axhline(1, c='k', lw=0.6)
      ax.errorbar(x, skm.real, ske.real, ls='', marker='.')
  elif obs_type in ['nofk']:
    # interpret
    nkm, nke = ym[0], ye[0]
    kvecs = np.asarray(meta['kvecs'])
    kmags = np.linalg.norm(kvecs, axis=-1)
    kf = 2./rs/nspin**0.5
    x = kmags/kf
    # visualize
    fig, ax = plt.subplots(1, 1)
    ax.axhline(0, c='k')
    ax.set_xlabel(r'$k/k_F$')
    ax.set_xlim(0, x.max())
    ax.set_ylabel('n(k)')

    ax.errorbar(x, nkm, nke, ls='', marker='.')
  else:
    msg = 'no "%s"' % obs_type
    raise RuntimeError(msg)

  fig.tight_layout()
  if args.savefig:
    fig_loc = f"plot_{obs_type}.pdf"
    fig.savefig(fig_loc)
  else:
    plt.show()

if __name__ == '__main__':
  main()  # set no global variable
