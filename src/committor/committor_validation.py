import numpy as np

################################################################################
#                Committor analysis: isocommittor curve and its points          #
################################################################################

# Problem-independent pieces of the committor analysis (Project 2, problem 1.2; Li et al.
# 2019): extract the level set q = 1/2 of a committor reduced to a 2D CV space, and sample
# starting points on it from the equilibrium density.
#
# The committor is reduced to CV space either by slicing (fix the other coordinates) or
# by projecting (average over them, projected_committor). Either way the caller provides
#   lift(s, z): (..., m), (..., d-m) -> (..., d), a point x with cv(x) = s;
# and, for the projection, fiber samples z of e^{-beta V} conditioned on cv(x) = s.
# Splitting the conditional sampler into fixed noise z and a deterministic lift lets every
# grid point use the same K draws (common random numbers), so the projected committor is
# a smooth function of s and its contour isn't jagged by Monte Carlo noise.


def projected_committor(q, s, *, lift, z, chunk=4096):
    '''
    Projected committor qbar(s) = E[q(x) | cv(x) = s], averaged over the K fiber samples z.

    q: (..., d) -> (...). s: (..., m). z: (K, d-m). Returns (...).
    '''
    s_flat= s.reshape(-1, s.shape[-1])
    qbar= np.concatenate([q(lift(s_c[:, None, :], z)).mean(axis=-1)     # (chunk, K, d) -> (chunk,)
                          for s_c in np.split(s_flat, np.arange(chunk, len(s_flat), chunk))])
    return qbar.reshape(s.shape[:-1])


def level_curve(s1, s2, f, level):
    '''
    Longest branch of the level set {f = level} of f sampled on the grid s1 x s2.

    s1: (n1,), s2: (n2,), f: (n2, n1) (meshgrid 'xy' indexing). Returns a polyline (L, 2).
    '''
    import contourpy
    lines= contourpy.contour_generator(s1, s2, f).lines(level)
    if not lines:
        raise ValueError(f'no level set at {level} on this grid')
    return max(lines, key=len)


def resample_arclength(curve, n):
    '''
    n points spaced uniformly in arc length along the polyline curve (L, m). Returns (n, m).
    '''
    seg= np.linalg.norm(np.diff(curve, axis=0), axis=-1)
    t= np.concatenate([[0], np.cumsum(seg)])
    t_new= np.linspace(0, t[-1], n)
    return np.stack([np.interp(t_new, t, curve[:, j]) for j in range(curve.shape[-1])], axis=-1)


def sample_on_curve(curve, *, log_density, N, rng, n_fine=10_000):
    '''
    N points on the polyline curve (L, m) drawn from the density exp(log_density(s))
    with respect to arc length (with replacement, from n_fine uniformly spaced points).

    log_density: (..., m) -> (...). Returns (N, m).
    '''
    s= resample_arclength(curve, n_fine)
    lw= log_density(s)
    w= np.exp(lw - lw.max())
    return s[rng.choice(n_fine, size=N, p=w / w.sum())]

def shooting_committor(points, *, M, dV, beta, h, ind_A, ind_B, rng, max_steps, progress_interval=None):
    '''
    Given points, launch M trajectories according to the overdamped Langevin dynamics
    from each until they arrive in
    either A or B, then returns M_A, and M_B, the count at A and B.
    Some trajectories might not arrive, so M_A + M_B <= M.

    points:   (..., d)
    dV:       (..., d) -> (..., d)
    ind_A,B:  (..., d) -> (...) (boolean)
    return:   (...), (...)
    '''
    from metadynamics import _euler_maruyama_step as step

    points= np.asarray(points, dtype=float)

    x= np.repeat(points[..., None, :], repeats=M, axis=-2)

    # trajectories which are still active
    # x.shape[:-1] since we don't need the dimension
    mask= np.ones(x.shape[:-1], dtype=bool)
    hit_A= np.zeros(x.shape[:-1], dtype=bool)
    hit_B= np.zeros(x.shape[:-1], dtype=bool)

    a= lambda t, x: -dV(x)
    b= lambda t, x: np.sqrt(2/beta)

    for N in range(max_steps):

        if progress_interval is not None:
            if N % max(1, max_steps // progress_interval) == 0:
                print(f'Step {N:06d}/{max_steps:06d}\t{np.sum(mask)} walkers remain')

        if not np.any(mask):
            break

        x[mask]= step(0, x[mask], a, b, h, rng)


        in_A= ind_A(x[mask])
        in_B= ind_B(x[mask])
        
        # these should have the same shape
        hit_A[mask] |= in_A
        hit_B[mask] |= in_B
        # we never need to update inactive trajectories
        mask[mask] &= ~(in_A | in_B)


    count_A= np.sum(hit_A, axis=-1)
    count_B= np.sum(hit_B, axis=-1)

    return count_A, count_B