'''
Faster drop-in versions of the slow parts of metadynamics.py.

Same function names and signatures, so a script only needs
`import metadynamics_optimized as md`. What changed:

- metadynamics_sampling: the frozen bias gradient is tabulated once on a grid in
  CV space and interpolated, instead of summing over all bumps at every step.
- delta_net: neighbors within delta are found with a KD-tree instead of computing
  the distance to every point.

Everything else (euler_maruyama, metadynamics_deposition, the exact bias
functions, initialize_walkers) is re-exported unchanged from metadynamics.py.
'''
import itertools

import numpy as np
from scipy.spatial import cKDTree

from metadynamics import (DEFAULT_PARAMS, euler_maruyama, gaussian_bias, gaussian_bias_grad,
                          initialize_walkers, metadynamics_deposition)

################################################################################
#                             Tabulated bias                                   #
################################################################################

# bumps are cut off beyond this many widths; exp(-6^2 / 2) ~ 1.5e-8
_CUTOFF= 6

def tabulate_bias(*, s_centers, height, width, dx):
    '''
    Tabulate the total gaussian bias and its gradient (both in CV space) on a
    uniform grid of spacing dx, covering the centers plus a margin of _CUTOFF widths.

    s_centers is (k, m), already in CV space. The grid has about (extent / dx)^m
    nodes, so this is only practical for small m (2, perhaps 3).

    Returns a dict with
        lo:   (m,)           coordinates of node 0
        dx:   float          grid spacing
        bias: (n_1..n_m)     V_G at the nodes
        grad: (n_1..n_m, m)  grad_s V_G at the nodes
    '''
    s_centers= np.asarray(s_centers, dtype=float)
    m= s_centers.shape[-1]
    lo= s_centers.min(axis=0) - _CUTOFF * width
    hi= s_centers.max(axis=0) + _CUTOFF * width
    n= np.ceil((hi - lo) / dx).astype(int) + 1

    bias= np.zeros(tuple(n))
    grad= np.zeros(tuple(n) + (m,))

    # each bump only touches the nodes within _CUTOFF widths of its center,
    # so add it on that patch alone
    r= int(np.ceil(_CUTOFF * width / dx))
    for c in s_centers:
        i_c= np.round((c - lo) / dx).astype(int)                      # nearest node
        i_lo= np.maximum(i_c - r, 0)
        i_hi= np.minimum(i_c + r + 1, n)
        patch= tuple(slice(a, b) for a, b in zip(i_lo, i_hi))
        axes= [lo[j] + dx * np.arange(i_lo[j], i_hi[j]) for j in range(m)]
        nodes= np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)   # (p_1..p_m, m)
        diffs= nodes - c                                              # (p_1..p_m, m)
        g= height * np.exp(-np.sum(diffs**2, axis=-1) / (2 * width**2))  # (p_1..p_m)
        bias[patch] += g
        grad[patch] += -g[..., None] * diffs / width**2

    return dict(lo=lo, dx=dx, bias=bias, grad=grad)

def _interpolate(table, s):
    '''
    Multilinear interpolation of `table` (n_1..n_m, ...) at points s (..., m),
    on the grid described by lo and dx. Points outside the grid get 0, which is
    correct up to the cutoff since every bump is negligible there.

    Returns (..., ) + table.shape[m:].
    '''
    lo, dx, values= table
    m= s.shape[-1]
    n= np.array(values.shape[:m])

    u= (s - lo) / dx                                                  # (..., m) position in grid units
    i= np.floor(u).astype(int)                                        # (..., m) lower node
    f= u - i                                                          # (..., m) fraction in [0, 1)
    inside= np.all((i >= 0) & (i < n - 1), axis=-1)                   # (...)
    i= np.clip(i, 0, n - 2)

    # sum over the 2^m corners of the cell, each weighted by prod_j (f_j or 1 - f_j)
    trailing= values.ndim - m
    result= 0
    for corner in itertools.product((0, 1), repeat=m):
        corner= np.array(corner)                                      # (m,)
        w= np.prod(np.where(corner, f, 1 - f), axis=-1)               # (...)
        idx= tuple((i + corner)[..., j] for j in range(m))            # m arrays of shape (...)
        result= result + w[(...,) + (None,) * trailing] * values[idx]
    return result * inside[(...,) + (None,) * trailing]

def grid_bias(*, x, grid, cv):
    '''
    Tabulated version of gaussian_bias. x is (..., d); returns (...).
    '''
    return _interpolate((grid['lo'], grid['dx'], grid['bias']), cv(x))

def grid_bias_grad(*, x, grid, cv, dcv):
    '''
    Tabulated version of gaussian_bias_grad. x is (..., d); returns (..., d).

    dcv is the transpose of the jacobian of the map from x to s
    '''
    ds= dcv(x)                                                        # (..., d, m)
    grad_s= _interpolate((grid['lo'], grid['dx'], grid['grad']), cv(x))  # (..., m)
    #         (..., d, m) @ (..., m)
    grad= np.matvec(ds, grad_s)                                       # (..., d)
    return grad

################################################################################
#                               Metadynamics                                   #
################################################################################

def metadynamics_sampling(*, N_samples, N_skip, dV, x_centers, cv, dcv, x0, rng, N_walkers=1, **param_overrides):
    '''
    Same as metadynamics.metadynamics_sampling, but with the bias tabulated on a
    grid in CV space (see tabulate_bias). The grid spacing is the parameter
    `grid_dx`, by default bump_width / 10.

    To evaluate the same bias afterwards (e.g. for weights e^{beta V_G}), call
    tabulate_bias with the same arguments and use grid_bias.
    '''
    params= DEFAULT_PARAMS | param_overrides
    bump_height= params['bump_height']
    bump_width= params['bump_width']
    h= params['h']
    beta= params['beta']
    grid_dx= params.get('grid_dx', bump_width / 10)

    x0= np.asarray(x0, dtype=float)
    d= x0.shape[-1]
    if x0.shape not in ((N_walkers, d), (d,)):
        raise TypeError('x0 must be of shape (N_walkers, d) or (d,)')
    x0= np.broadcast_to(x0, (N_walkers, d)).copy()

    grid= tabulate_bias(s_centers=cv(x_centers), height=bump_height, width=bump_width, dx=grid_dx)

    a= lambda t, x: -(dV(x) + grid_bias_grad(x=x, grid=grid, cv=cv, dcv=dcv))
    b= lambda t, x: np.sqrt(2 / beta)
    N_per_walker= -(-N_samples // N_walkers)                           # ceil
    N_steps= N_skip * N_per_walker
    _, x= euler_maruyama(0, x0, a, b, h, N_steps=N_steps, N_skip=N_skip, rng=rng)  # (N_per_walker + 1, N_walkers, d)

    # drop x0, then trim the excess from rounding up
    x_flat= x[1:].reshape(-1, d)[:N_samples]                          # (N_samples, d)

    return x_flat

def delta_net(x, delta):
    '''
    Given x of shape (n, d), perform the delta net algorithm to thin it.
    Returns the mask of kept points.

    Same greedy algorithm as metadynamics.delta_net, but the points within delta
    of x[k] come from a KD-tree query instead of distances to all n points.
    (The query includes distance exactly delta; this has no practical effect.)
    '''
    x= np.asarray(x, dtype=float)
    n, d= x.shape

    tree= cKDTree(x)
    keep= np.zeros(n, dtype=bool)
    discarded= np.zeros(n, dtype=bool)

    for k in range(n):
        if discarded[k]:
            continue
        keep[k]= True
        discarded[tree.query_ball_point(x[k], delta)]= True

    return keep
