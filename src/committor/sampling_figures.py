import numpy as np
import matplotlib.pyplot as plt
from mueller import V
from pathlib import Path

ROOT= Path(__file__).resolve().parents[2]

x= np.load(ROOT / 'data' / 'committor' / 'training_data_31415926535897932384626433.npy')
padding=0.25

xmin= x[:,0].min()-padding
xmax= x[:,0].max()+padding
ymin= x[:,1].min()-padding
ymax= x[:,1].max()+padding


# compute the potential on a grid
grid_size= 500

x1_grid= np.linspace(xmin, xmax, grid_size)
x2_grid= np.linspace(ymin, ymax, grid_size)
X1_grid, X2_grid= np.meshgrid(x1_grid, x2_grid)
X_grid= np.stack([X1_grid, X2_grid], axis=-1)
V_grid= V(X_grid)
V_grid= np.clip(V_grid, -500, 200)


fig, ax= plt.subplots(figsize=(10, 8), dpi=300)
ax.set_aspect('equal')
ax.set(
    title='Training points and V(x)',
    xlabel='$x_1$',
    ylabel='$x_2$',
)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

ax.scatter(x[:,0], x[:,1], s=3, c='k', alpha=0.1)
cntr= ax.contour(X_grid[...,0], X_grid[...,1], V_grid, levels=15)
fig.colorbar(cntr, ax=ax)

fig.savefig(ROOT / 'writeup' / 'figures' / 'p1_sample_points.png')