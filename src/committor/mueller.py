import numpy as np

MUELLER_SEED= 31415926535897932384626433
# MUELLER_SEED= 42

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

# metastable sets A and B: cylinders of radius R around these points in (x1, x2)
# (Yuan et al. 2023, Sec. 6.2, following Li et al. 2019 Sec. IV.A)
CENTER_A = np.array([-0.558, 1.441])
CENTER_B = np.array([0.623, 0.028])
R = 0.1

# metadynamics settings (Yuan et al. 2023 Sec. 6.1.1, "settings used in" Li et al. 2019 Sec. IV.A)
PARAMS = dict(
    Nskip= 500,
    Nbumps= 2000,
    bump_height= 5,
    bump_width= 0.05,
    h= 1e-5,
    beta= 0.1,
)


def _mueller_terms(x):
    # x has shape (..., d); returns the offsets and the four exponential terms, each of shape (..., 4)
    dx = x[..., 0, None] - _X
    dy = x[..., 1, None] - _Y
    E = _D * np.exp(_A * dx**2 + _B * dx * dy + _C * dy**2)
    return dx, dy, E


def V(x, gamma=GAMMA, k=K, sigma=SIGMA):
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


def dV(x, gamma=GAMMA, k=K, sigma=SIGMA):
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

def cv(x):
    '''
    Collective variable map for the rugged Mueller potential
    '''
    return x[...,:2]
def dcv(x):
    '''
    (tranpose) Jacobian of the collective variable map for rugged Mueller potential
    '''
    d= x.shape[-1]
    return np.eye(d)[:, :2]
def in_A(x):
    '''
    Mask of the points of x (..., d) in the set A. Returns (...).
    '''
    return np.linalg.norm(cv(x) - CENTER_A, axis=-1) < R
def in_B(x):
    '''
    Mask of the points of x (..., d) in the set B. Returns (...).
    '''
    return np.linalg.norm(cv(x) - CENTER_B, axis=-1) < R


def indicators(*, r, w):
    '''
    Smoothed indicators (chi_A, chi_B) of the sets A and B, as torch functions
    (..., d) -> (...) for the committor model. They are cylinders of radius r in CV
    space, smoothed over a width w outside them (see committor_model.chi).
    '''
    from committor_model import chi     # torch is only needed here, not for sampling
    chi_A= lambda x: chi(cv(x), c=CENTER_A, r=r, w=w)
    chi_B= lambda x: chi(cv(x), c=CENTER_B, r=r, w=w)
    return chi_A, chi_B
