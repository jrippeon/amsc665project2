import numpy as np
import mueller
import committor_model
import matplotlib.pyplot as plt
import torch

SEED= mueller.MUELLER_SEED

# for filepaths
from pathlib import Path
ROOT= Path(__file__).resolve().parents[2]

data_dir= ROOT / 'data' / 'committor'
ckpt_file= data_dir / f'committor_baseline_{SEED}.pt'

figure_dir= ROOT / 'writeup' / 'figures'
figure_prefix= 'p1'


model, ck= committor_model.load_committor(ckpt_file, indicators=mueller.indicators)

history= ck['history']
train_loss= history['train_loss']
test_loss= history['test_loss']

fig, ax= plt.subplots()

ax.semilogy(train_loss)
ax.semilogy(test_loss)
ax.legend(['train', 'test'])
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
fig.savefig(figure_dir / f'{figure_prefix}_training_history.eps')




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

fig, ax= plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.set(
    title='Committor $q(x_1, x_2, 0, \\dots, 0)$',
    xlabel='$x_1$',
    ylabel='$x_2$',
)
cntrf= ax.contourf(X1_grid, X2_grid, q_grid, levels=np.linspace(0, 1, 21), cmap='viridis')
fig.colorbar(cntrf, ax=ax, label='$q$')
ax.contour(X1_grid, X2_grid, V_grid, levels=15, colors='gray', linewidths=1, alpha=0.75)
for center, name in [(mueller.CENTER_A, 'A'), (mueller.CENTER_B, 'B')]:
    ax.add_patch(plt.Circle(center, mueller.R, fill=False, color='r'))
    ax.annotate(name, center, color='r', ha='center', va='center')
ax.scatter(x_train[:,0], x_train[:,1], s=1, c='r', alpha=0.1, linewidths=0, rasterized=True)
fig.savefig(figure_dir / f'{figure_prefix}_committor.png', dpi=300)
