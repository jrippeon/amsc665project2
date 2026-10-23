import numpy as np


################################################################################
#                             Mueller Potential                                #
################################################################################

# Mueller potential parameters (Yuan et al. 2023, eqn (71))
_A = np.array([-1.0, -1.0, -6.5, 0.7])
_B = np.array([0.0, 0.0, 11.0, 0.6])
_C = np.array([-10.0, -10.0, -6.5, 0.7])
_D = np.array([-200.0, -100.0, -170.0, 15.0])
_X = np.array([1.0, 0.0, -0.5, -1.0])
_Y = np.array([0.0, 0.5, 1.5, 1.0])

# rugged perturbation and harmonic confinement (Yuan et al. 2023, eqn (82))
GAMMA = 9.0
K = 5.0
SIGMA = 0.05


def _mueller_terms(x):
    # x has shape (..., d); returns the offsets and the four exponential terms, each of shape (..., 4)
    dx = x[..., 0, None] - _X
    dy = x[..., 1, None] - _Y
    E = _D * np.exp(_A * dx**2 + _B * dx * dy + _C * dy**2)
    return dx, dy, E


def V_mueller(x, gamma=GAMMA, k=K, sigma=SIGMA):
    """Rugged Mueller potential, eqn (82) of Yuan et al. (2023).

    V(x) = V_0(x1, x2) + gamma sin(2 k pi x1) sin(2 k pi x2) + (1/sigma^2) sum_{i>=3} x_i^2

    x: array of shape (..., d) with d >= 2. Returns an array of shape (...).
    """
    x = np.asarray(x, dtype=float)
    _, _, E = _mueller_terms(x)
    w = 2.0 * k * np.pi
    V = E.sum(axis=-1)
    V += gamma * np.sin(w * x[..., 0]) * np.sin(w * x[..., 1])
    V += np.sum(x[..., 2:]**2, axis=-1) / sigma**2
    return V


def dV_mueller(x, gamma=GAMMA, k=K, sigma=SIGMA):
    """Gradient of the rugged Mueller potential V_mueller.

    x: array of shape (..., d) with d >= 2. Returns an array of shape (..., d).
    """
    x = np.asarray(x, dtype=float)
    dx, dy, E = _mueller_terms(x)
    w = 2.0 * k * np.pi
    s1, c1 = np.sin(w * x[..., 0]), np.cos(w * x[..., 0])
    s2, c2 = np.sin(w * x[..., 1]), np.cos(w * x[..., 1])

    grad = np.empty_like(x)
    grad[..., 0] = np.sum(E * (2.0 * _A * dx + _B * dy), axis=-1) + gamma * w * c1 * s2
    grad[..., 1] = np.sum(E * (_B * dx + 2.0 * _C * dy), axis=-1) + gamma * w * s1 * c2
    grad[..., 2:] = 2.0 * x[..., 2:] / sigma**2
    return grad

################################################################################
#                              Euler Maruyama                                  #
################################################################################

def _euler_maruyama_step(t, x, a, b, dt):
    '''
    One Euler-Maruyama step on the SDE

    dx = a(t, x) dt + b(t, x) dw

    with timestep dt
    '''
    x= np.asarray(x, dtype=float)
    w= np.random.normal(0, np.sqrt(dt), x.shape)
    x_next= x + a(t, x) * dt + b(t, x) * w
    return x_next

def euler_maruyama(t0, x0, a, b, dt, T=None, N_steps=None):
    '''
    Run an Euler-Maruyama simulation of the SDE

    dx = a(t, x) dt + b(t, x) dw

    on time [t0, T] starting from x0, with timestep dt
    '''
    x0= np.asarray(x0, dtype=float)
    dim= x0.shape

    if T is not None and N_steps is None:
        t= np.arange(t0, T, dt)
        N_steps= t.shape[0]
    elif T is None and N_steps is not None:
        # t= np.arange(t0, t0+N_steps*dt, dt)
        t= t0 + dt * np.arange(N_steps)
    else:
        raise TypeError('Exactly one of T and N_steps must be provided.')

    x= np.zeros((N_steps,) + dim)
    x[0]= x0
    for n in range(N_steps-1):
        x[n+1]= _euler_maruyama_step(t[n], x[n], a, b, dt)

    return t, x

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
    

def V_G_mueller(x, x_bump, bump_width=DEFAULT_PARAMS['bump_width'], bump_height=DEFAULT_PARAMS['bump_height']):
    # in the Mueller potential only the first two coordinates are used as "collective variables"
    diff= x[...,:2] - x_bump[...,:2]
    bump= bump_height * np.exp(-np.sum(diff**2, axis=-1) / (2 * bump_width**2))
    return bump

def dV_G_mueller(x, x_bump, bump_width=DEFAULT_PARAMS['bump_width'], bump_height=DEFAULT_PARAMS['bump_height']):
    # (..., 2)
    diff= x[...,:2] - x_bump[...,:2]
    # (...)
    g= -bump_height / bump_width**2 * np.exp(-np.sum(diff**2, axis=-1) / (2 * bump_width**2))
    #         (..., 1)   (..., 2) 
    bump= g[..., None] * diff
    # x and x_bump broadcast against each other (e.g. (..., 1, d) and (k, d) give (..., k, d)),
    # so the output must have the broadcast shape rather than the shape of x
    result= np.zeros(np.broadcast_shapes(x.shape, x_bump.shape))
    result[..., :2]= bump
    return result




def metadynamics_sample(*, dV, dV_G, x0=None, d=None, **param_overrides):
    '''
    Take a sample of N = Nskip * Nbumps points from the potential well with gradient dV,
    using the metadynamics algorithm with bump function dV_G.

    Returns (N, d)  array of points in R^d.
    '''
    params= DEFAULT_PARAMS | param_overrides
    Nskip= params['Nskip']
    Nbumps= params['Nbumps']
    # bump_height= params['bump_height']
    # bump_width= params['bump_width']
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
    x_bumps= np.zeros((Nbumps, d))
    x[0]= x0

    b_langevin= lambda t, x: np.sqrt(2 / beta)

    for k in range(Nbumps):

        if k % 10 == 0:
            print(f'{k=}')


        # for the zeroth we do nothing since we don't have any bumps yet
        if k > 0:
            x0= x[k*Nskip-1]
            # x is (..., d), and x_bumps is (k, d), so this is (..., k, d)
            # (recall broadcasting works right-to-left)
            a_langevin= lambda t, x: -(dV(x) + np.sum(
                dV_G(x=x[..., None, :], x_bump=x_bumps[:k, :], bump_width=params['bump_width'], bump_height=params['bump_height']),
                axis=-2))
        else:
            a_langevin= lambda t, x: -dV(x)

        # we don't care about time at all for this sampling method
        _, new_xs= euler_maruyama(t0=0, x0=x0, a=a_langevin, b=b_langevin, dt=h, N_steps=Nskip+1)
        # the first step is x0, we discard it
        x[k*Nskip:(k+1)*Nskip]= new_xs[1:]
        x_bumps[k]= new_xs[-1]

    return x