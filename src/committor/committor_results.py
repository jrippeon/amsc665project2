import numpy as np
import mueller
import committor_model
import committor_validation as validation
import matplotlib.pyplot as plt
import torch

SEED= mueller.MUELLER_SEED

# for filepaths
from pathlib import Path
ROOT= Path(__file__).resolve().parents[2]

data_dir= ROOT / 'data' / 'committor'
RUN_NAME= 'baseline'     # the training run to analyse, as in train_model.py
ckpt_file= data_dir / f'committor_{RUN_NAME}_{SEED}.pt'

figure_dir= ROOT / 'writeup' / 'figures'
figure_prefix= f'p1_{RUN_NAME}'


model, ck= committor_model.load_committor(ckpt_file, indicators=mueller.indicators)

history= ck['history']
train_loss= history['train_loss']
test_loss= history['test_loss']


### Plot training history ###

fig, ax= plt.subplots()

ax.semilogy(train_loss)
ax.semilogy(test_loss)
ax.legend(['train', 'test'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'{RUN_NAME} training history')
fig.savefig(figure_dir / f'{figure_prefix}_training_history.eps')
fig.savefig(figure_dir / f'{figure_prefix}_training_history.png')



### Evaluate committor on a grid ###

# committor on the (x1, x2) plane, with x3 = ... = xd = 0
q= committor_model.as_numpy(model)
d= ck['model_params']['d']

x_train= np.load(data_dir / f'training_data_{SEED}.npy')
padding= 0.25
x1_grid= np.linspace(x_train[:,0].min()-padding, x_train[:,0].max()+padding, 500)
x2_grid= np.linspace(x_train[:,1].min()-padding, x_train[:,1].max()+padding, 500)
X1_grid, X2_grid= np.meshgrid(x1_grid, x2_grid)
X_grid= np.zeros(X1_grid.shape + (d,))
X_grid[...,0]= X1_grid
X_grid[...,1]= X2_grid

q_grid= q(X_grid)
V_grid= np.clip(mueller.V(X_grid), -500, 200)

# away from the training points q is extrapolated, so only that region is coloured
support_radius= 0.05    # well above the typical training point spacing (~0.006, 99th percentile ~0.02)
in_support= validation.near_points(X_grid[..., :2], x_train[:, :2], radius=support_radius)
q_shown= np.ma.masked_where(~in_support, q_grid)

fig, ax= plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.set(
    title=f'Committor $q(x_1, x_2, 0, \\dots, 0)$ within {support_radius} of the training points',
    xlabel='$x_1$',
    ylabel='$x_2$',
)
cntrf= ax.contourf(X1_grid, X2_grid, q_shown, levels=np.linspace(0, 1, 21), cmap='viridis')
fig.colorbar(cntrf, ax=ax, label='$q$')
ax.contour(X1_grid, X2_grid, V_grid, levels=15, colors='gray', linewidths=1, alpha=0.75)
for center, name in [(mueller.CENTER_A, 'A'), (mueller.CENTER_B, 'B')]:
    ax.add_patch(plt.Circle(center, mueller.R, fill=False, color='r'))
    ax.annotate(name, center, color='r', ha='center', va='center')
ax.scatter(x_train[:,0], x_train[:,1], s=1, c='r', alpha=0.1, linewidths=0, rasterized=True)
fig.savefig(figure_dir / f'{figure_prefix}_committor.png', dpi=300)



### Isocommittor curve q = 1/2 and points sampled on it ###

# Committor analysis (Project 2, problem 1.2): extract the level set q = 1/2 of the
# committor on the slice x3 = ... = xd = 0, and sample starting points on it from e^{-beta V}.
#
# Slicing is enough here: V separates into a part in (x1, x2) and independent harmonic
# terms in x3, ..., xd, so the dynamics of (x1, x2) don't see the other coordinates, and A
# and B depend only on (x1, x2). The exact committor is then independent of x3, ..., xd.
# For the same reason the (x1, x2)-marginal of e^{-beta V} is exp(-beta V(x1, x2, 0, ..., 0)),
# and conditioned on (x1, x2) the x_i, i >= 3, are i.i.d. N(0, sigma^2/(2 beta)).
# TODO: lift, log_density and the fiber distribution are problem-specific, belong in mueller.py (rule 1).
beta= mueller.PARAMS['beta']
s_perp= mueller.SIGMA / np.sqrt(2 * beta)

def lift(s, z):
    batch= np.broadcast_shapes(s.shape[:-1], z.shape[:-1])
    return np.concatenate([np.broadcast_to(s, batch + s.shape[-1:]),
                           np.broadcast_to(z, batch + z.shape[-1:])], axis=-1)

def log_density(s):
    return -beta * mueller.V(lift(s, np.zeros(d - 2)))


### sample starting points ###

N_iso= 100      # starting points on the curve (Project 2: N = 50 or 100)

rng= np.random.default_rng(SEED)

curve= validation.level_curve(x1_grid, x2_grid, q_grid, 0.5)
s_iso= validation.sample_on_curve(curve, log_density=log_density, N=N_iso, rng=rng)
x_iso= lift(s_iso, rng.normal(scale=s_perp, size=(N_iso, d - 2)))           # (N_iso, d)
np.save(data_dir / f'isocommittor_points_{RUN_NAME}_{SEED}.npy', x_iso)

### estimate the committor ###
M=200
max_steps= 50_000

# the shooting takes about a minute, so its counts are saved and reused while the
# starting points and settings are unchanged
shooting_file= data_dir / f'shooting_{RUN_NAME}_{SEED}.npz'
saved= np.load(shooting_file) if shooting_file.exists() else None
if (saved is not None and np.array_equal(saved['x_iso'], x_iso)
        and saved['M'] == M and saved['max_steps'] == max_steps):
    M_A, M_B= saved['M_A'], saved['M_B']
else:
    M_A, M_B= validation.shooting_committor(x_iso,
                                        M=M,
                                        dV=mueller.dV,
                                        beta=mueller.PARAMS['beta'],
                                        h=mueller.PARAMS['h'],
                                        ind_A=mueller.in_A,
                                        ind_B=mueller.in_B,
                                        rng=rng,
                                        max_steps=max_steps,
                                        progress_interval=10,
                                        )
    np.savez(shooting_file, x_iso=x_iso, M=M, max_steps=max_steps, M_A=M_A, M_B=M_B)
# walkers that didn't finish within max_steps are left out of the estimate
M_done= M_A + M_B
print(f'{np.sum(M - M_done)} of {N_iso * M} walkers did not reach A or B')
q_shooting= M_B / M_done

### plot sampled starting points ###

fig, ax= plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.set(
    title='Isocommittor curve $q = 1/2$ and sampled starting points',
    xlabel='$x_1$',
    ylabel='$x_2$',
)
cntrf= ax.contourf(X1_grid, X2_grid, q_shown, levels=np.linspace(0, 1, 21), cmap='viridis')
fig.colorbar(cntrf, ax=ax, label='$q$')
ax.contour(X1_grid, X2_grid, V_grid, levels=15, colors='gray', linewidths=1, alpha=0.75)
for center, name in [(mueller.CENTER_A, 'A'), (mueller.CENTER_B, 'B')]:
    ax.add_patch(plt.Circle(center, mueller.R, fill=False, color='r'))
    ax.annotate(name, center, color='r', ha='center', va='center')
# only the part of the curve in the coloured region
curve_shown= np.where(validation.near_points(curve, x_train[:, :2], radius=support_radius)[:, None],
                      curve, np.nan)
ax.plot(curve_shown[:,0], curve_shown[:,1], 'w', lw=1.5, label='$q(x_1, x_2, 0, \\dots, 0) = 1/2$')
ax.scatter(s_iso[:,0], s_iso[:,1], s=12, c='r', zorder=3, label=f'{N_iso} starting points')
ax.legend(loc='upper right', facecolor='lightgray')
fig.savefig(figure_dir / f'{figure_prefix}_isocommittor.png', dpi=300)


### histogram of the shooting committor ###

# binomial standard error of each estimate at p = 1/2
se= 0.5 / np.sqrt(M)
print(f'shooting committor: mean {q_shooting.mean():.3f}, median {np.median(q_shooting):.3f}, '
      f'std {q_shooting.std():.3f} (binomial noise alone: {se:.3f})')

fig, ax= plt.subplots()
ax.hist(q_shooting, bins=np.linspace(0, 1, 21), color='C0', edgecolor='k')
ax.axvline(0.5, color='k', ls='--', label='$p = 1/2$')
ax.axvline(np.mean(q_shooting), color='r', label=f'mean $= {np.mean(q_shooting):.3f}$')
ax.set(
    title=f'Committor analysis: {N_iso} points on $q = 1/2$, {M} trajectories each',
    xlabel='$p$ (fraction reaching B before A)',
    ylabel='count',
)
ax.legend()
fig.savefig(figure_dir / f'{figure_prefix}_committor_histogram.eps')
fig.savefig(figure_dir / f'{figure_prefix}_committor_histogram.png')
