import os

import matplotlib.pyplot as plt
import numpy as np

from timeseries.data.lorenz.lorenz import Lorenz

save_folder = 'images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

start_time = 0
end_time = 100
time_points = np.linspace(start_time, end_time, end_time * 15)

lorenz_sys = Lorenz(sigma=10., rho=28., beta=8. / 3.)
x, y, z = lorenz_sys.solve(time_points)

# plot the lorenz attractor in three-dimensional phase space
fig = plt.figure(figsize=(12, 9))
ax = fig.gca(projection='3d')
ax.xaxis.set_pane_color((1, 1, 1, 1))
ax.yaxis.set_pane_color((1, 1, 1, 1))
ax.zaxis.set_pane_color((1, 1, 1, 1))
ax.plot(x, y, z, color='g', alpha=0.7, linewidth=0.6)
ax.set_title('Lorenz attractor phase diagram')

fig.savefig('{}/lorenz-attractor-3d.png'.format(save_folder), dpi=180, bbox_inches='tight')
plt.show()

# now plot two-dimensional cuts of the three-dimensional phase space
fig, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(17, 6))

# plot the x values vs the y values
ax[0].plot(x, y, color='r', alpha=0.7, linewidth=0.3)
ax[0].set_title('x-y phase plane')

# plot the x values vs the z values
ax[1].plot(x, z, color='m', alpha=0.7, linewidth=0.3)
ax[1].set_title('x-z phase plane')

# plot the y values vs the z values
ax[2].plot(y, z, color='b', alpha=0.7, linewidth=0.3)
ax[2].set_title('y-z phase plane')

fig.savefig('{}/lorenz-attractor-phase-plane.png'.format(save_folder), dpi=180, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(3, 1, figsize=(17, 10))
# plot the x values vs the y values
ax[0].plot(time_points, x, color='r', alpha=1, linewidth=0.4)
ax[0].set_title('x phase plane')

# plot the x values vs the z values
ax[1].plot(time_points, y, color='m', alpha=1, linewidth=0.4)
ax[1].set_title('y phase plane')

# plot the y values vs the z values
ax[2].plot(time_points, z, color='b', alpha=1, linewidth=0.4)
ax[2].set_title('zz phase plane')

fig.savefig('{}/lorenz-attractor-time-series.png'.format(save_folder), dpi=180, bbox_inches='tight')
plt.show()