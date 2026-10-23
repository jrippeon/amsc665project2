import numpy as np

################################################################################
#                              Euler Maruyama                                  #
################################################################################

def _euler_maruyama_step(t, x, a, b, dt, rng):
    '''
    One Euler-Maruyama step on the SDE

    dx = a(t, x) dt + b(t, x) dw

    with timestep dt, drawing the noise from the np.random.Generator rng
    '''
    x= np.asarray(x, dtype=float)
    w= rng.normal(0, np.sqrt(dt), x.shape)
    x_next= x + a(t, x) * dt + b(t, x) * w
    return x_next

def euler_maruyama(t0, x0, a, b, dt, T=None, N_steps=None, N_skip=1, *, rng):
    '''
    Run an Euler-Maruyama simulation of the SDE

    dx = a(t, x) dt + b(t, x) dw

    on time [t0, T] starting from x0, with timestep dt.
    N_steps is the number of steps taken.
    rng is an np.random.Generator, advanced in place.

    Returns x0 followed by every N_skip'th state, N_steps // N_skip + 1 samples.
    '''
    x0= np.asarray(x0, dtype=float) # (...)
    dim= x0.shape

    if (T is None) == (N_steps is None):
        raise TypeError('Exactly one of T and N_steps must be provided.')

    if T is not None:
        N_steps= round((T-t0) / dt)

    N_samples= N_steps // N_skip + 1                  # x0, then every N_skip'th state
    t_result= t0 + dt * N_skip * np.arange(N_samples)

    x_result= np.zeros((N_samples,) + dim)
    x_result[0]= x0
    x_curr= x0
    for n in range(N_steps):
        x_curr= _euler_maruyama_step(t0 + n * dt, x_curr, a, b, dt, rng)
        if (n + 1) % N_skip == 0:
            x_result[(n + 1) // N_skip]= x_curr

    return t_result, x_result

################################################################################
#                               Metadynamics                                   #
################################################################################

# parameters used in Yuan 2023 for the rugged Mueller problem
DEFAULT_PARAMS= dict(
    Nskip= 500,
    Nbumps= 2000,
    bump_height= 5,
    bump_width= 0.05, 
    h= 1e-5,
    beta= 0.1
)
    
def gaussian_bias(*,x, x_centers, height, width, cv):
    '''
    Given x of shape (..., d), and k center locations (k, d),
    compute the total gaussian bias at point x.
    '''
    s= cv(x)                                                          # (..., m)
    s_centers= cv(x_centers)                                          # (k, m)
    diffs= s[...,None,:] - s_centers                                  # (..., k, m)
    g= height * np.exp(-np.sum(diffs**2, axis=-1) / (2 * width**2))   # (..., k)
    bias= np.sum(g, axis=-1)                                          # (...,)
    return bias

def gaussian_bias_grad(*,x, x_centers, height, width, cv, dcv):
    '''
    Given x of shape (..., d), and k center locations (k, d),
    compute the total gradient of gaussian bias at point x.

    dcv is the transpose of the jacobian of the map from x to s
    '''
    s= cv(x)                                                                      # (..., m)
    ds= dcv(x)                                                                    # (..., d, m)
    s_centers= cv(x_centers)                                                      # (k, m)
    diffs= s[...,None,:] - s_centers                                              # (..., k, m)
    g= -height / width**2 * np.exp(-np.sum(diffs**2, axis=-1) / (2 * width**2))   # (..., k)
    #              (..., k, 1)   (..., k, m)
    grad_s= np.sum(g[..., None] * diffs, axis=-2)                                 # (..., m)
    #         (..., d, m) @ (..., m)
    grad= np.matvec(ds, grad_s)                                                   # (..., d)
    return grad


def metadynamics_deposition(*, dV, cv, dcv, rng, x0=None, d=None, **param_overrides):
    '''
    Perform the deposition phase of the metadynamics algorithm, placing Gaussians
    in the space of the collective variables.

    cv represents the map
    xi : R^d --> R^m from real variables to collective variables,
    and dcv is the transpose of the jacobian thereof.

    cv should take (..., d) to (..., m),
    and dcv should take (..., d) to (..., d, m)

    rng is an np.random.Generator, advanced in place.

    Returns (N, d)  array of points in R^d, where N = Nskip * Nbumps
    '''
    params= DEFAULT_PARAMS | param_overrides
    Nskip= params['Nskip']
    Nbumps= params['Nbumps']
    bump_height= params['bump_height']
    bump_width= params['bump_width']
    h= params['h']
    beta= params['beta']

    if (x0 is None) and (d is None):
        raise TypeError('Exactly one of x0 and d must be provided (or we can\'t determine the dimesions of x)')
    elif x0 is not None:
        x0= np.asarray(x0, dtype=float)
        d= x0.shape[-1]
    elif d is not None:
        x0= np.zeros(d)

    N= Nskip * Nbumps
    x= np.zeros((N,d))
    x_centers= np.zeros((Nbumps, d))
    x[0]= x0

    b_langevin= lambda t, x: np.sqrt(2 / beta)

    for k in range(Nbumps):

        if k % max(1, Nbumps // 20) == 0:
            print(f'{100*k//Nbumps}% \t {k=}')


        a_langevin= lambda t, x: -dV(x) - gaussian_bias_grad(x=x,
                                                             x_centers=x_centers[:k],
                                                             height=bump_height,
                                                             width=bump_width,
                                                             cv=cv,
                                                             dcv=dcv)
        # we don't care about time at all for this sampling method
        _, new_xs= euler_maruyama(t0=0, x0=x0, a=a_langevin, b=b_langevin, dt=h, N_steps=Nskip, rng=rng)
        # the first step is x0, we discard it
        x[k*Nskip:(k+1)*Nskip]= new_xs[1:]
        x_centers[k]= new_xs[-1]
        # update our starting position for next iteration
        x0= new_xs[-1]

    return x, x_centers

def metadynamics_sampling(*, N_samples, N_skip, dV, x_centers, cv, dcv, x0, rng, N_walkers=1, **param_overrides):
    '''
    Run `N_walkers` Euler-Maruyama trajectories in the potential landscape defined by
    V and bumps at `x_centers` with height `bump_height` and width `bump_width`.

    x0 should be of shape (N_walkers, d) or (d,) (in which case all walkers start in the same place).
    rng is an np.random.Generator, advanced in place.
    '''
    params= DEFAULT_PARAMS | param_overrides
    bump_height= params['bump_height']
    bump_width= params['bump_width']
    h= params['h']
    beta= params['beta']

    x0= np.asarray(x0, dtype=float)
    d= x0.shape[-1]
    if x0.shape not in ((N_walkers, d), (d,)):
        raise TypeError('x0 must be of shape (N_walkers, d) or (d,)')
    x0= np.broadcast_to(x0, (N_walkers, d)).copy()                    # (N_walkers, d)

    a= lambda t, x: -(dV(x) + gaussian_bias_grad(x=x, x_centers=x_centers, height=bump_height, width=bump_width, cv=cv, dcv=dcv))
    b= lambda t, x: np.sqrt(2 / beta)
    N_per_walker= -(-N_samples // N_walkers)                           # ceil
    N_steps= N_skip * N_per_walker
    _, x= euler_maruyama(0, x0, a, b, h, N_steps=N_steps, N_skip=N_skip, rng=rng)  # (N_per_walker + 1, N_walkers, d)

    # drop x0, then trim the excess from rounding up
    x_flat= x[1:].reshape(-1, d)[:N_samples]                          # (N_samples, d)

    return x_flat

def initialize_walkers(x_centers, N_walkers, *, rng):
    '''
    Distribute the initial walkers among the centers of the bumps.
    This should lead to a reasonable distribution, with some in each basin.

    x_centers is (k, d); returns (N_walkers, d). Centers are chosen without
    replacement unless N_walkers > k.
    '''
    k= x_centers.shape[0]
    idx= rng.choice(k, N_walkers, replace=N_walkers > k)              # (N_walkers,)
    return x_centers[idx]