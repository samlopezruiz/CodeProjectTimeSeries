import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os


class Lorenz:

    def __init__(self, sigma=10., rho=-28., beta=8. / 3.):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.initial_state = [0, 1, 1.05]
        start_time = 0
        end_time = 100
        self.x = self.y = self.z = [0]
        self.time_points = np.linspace(start_time, end_time, end_time * 100)

    def lorenz_system(self, current_state, t):
        # positions of x, y, z in space at the current time point
        x, y, z = current_state

        # define the 3 ordinary differential equations known as the lorenz equations
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z

        # return a list of the equations that describe the system
        return [dx_dt, dy_dt, dz_dt]

    def solve(self, time_points=None):
        if time_points is None:
            time_points = self.time_points
        else:
            self.time_points = time_points
        xyz = odeint(self.lorenz_system, self.initial_state, time_points)

        # extract the individual arrays of x, y, and z values from the array of arrays
        self.x = xyz[:, 0]
        self.y = xyz[:, 1]
        self.z = xyz[:, 2]

    def plot3d(self, save_folder=None):
        if len(self.x) > 1:
            # plot the lorenz attractor in three-dimensional phase space
            fig = plt.figure(figsize=(12, 9))
            ax = fig.gca(projection='3d')
            ax.xaxis.set_pane_color((1, 1, 1, 1))
            ax.yaxis.set_pane_color((1, 1, 1, 1))
            ax.zaxis.set_pane_color((1, 1, 1, 1))
            ax.plot(self.x, self.y, self.z, color='g', alpha=0.7, linewidth=0.6)
            ax.set_title('Lorenz attractor phase diagram')

            if save_folder is not None:
                fig.savefig('{}/lorenz-attractor-3d.png'.format(save_folder), dpi=180, bbox_inches='tight')
            plt.show()
        else:
            print("Solve ODE first")

    def plot2d(self, save_folder=None):
        if len(self.x) > 1:
            # now plot two-dimensional cuts of the three-dimensional phase space
            fig, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(17, 6))

            # plot the x values vs the y values
            ax[0].plot(self.x, self.y, color='r', alpha=0.7, linewidth=0.3)
            ax[0].set_title('x-y phase plane')

            # plot the x values vs the z values
            ax[1].plot(self.x, self.z, color='m', alpha=0.7, linewidth=0.3)
            ax[1].set_title('x-z phase plane')

            # plot the y values vs the z values
            ax[2].plot(self.y, self.z, color='b', alpha=0.7, linewidth=0.3)
            ax[2].set_title('y-z phase plane')

            if save_folder is not None:
                fig.savefig('{}/lorenz-attractor-phase-plane.png'.format(save_folder), dpi=180, bbox_inches='tight')
            plt.show()
        else:
            print("Solve ODE first")

    def plot_time_series(self, save_folder=None):
        if len(self.x) > 1:
            fig, ax = plt.subplots(3, 1, figsize=(17, 10))
            # plot the x values vs the y values
            ax[0].plot(self.time_points, self.x, color='r', alpha=1, linewidth=0.4)
            ax[0].set_title('x phase plane')

            # plot the x values vs the z values
            ax[1].plot(self.time_points, self.y, color='m', alpha=1, linewidth=0.4)
            ax[1].set_title('y phase plane')

            # plot the y values vs the z values
            ax[2].plot(self.time_points, self.z, color='b', alpha=1, linewidth=0.4)
            ax[2].set_title('zz phase plane')

            if save_folder is not None:
                fig.savefig('{}/lorenz-attractor-time-series.png'.format(save_folder), dpi=180, bbox_inches='tight')
            plt.show()
        else:
            print("Solve ODE first")

    def get_time_series(self):
        return [self.x, self.y, self.z]


if __name__ == '__main__':
    save_folder = 'images'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    start_time = 0
    end_time = 100
    t = np.linspace(start_time, end_time, end_time * 100)

    lorenz_sys = Lorenz(sigma=10., rho=28., beta=8. / 3.)
    lorenz_sys.solve(t)
    xyz = lorenz_sys.get_time_series()
    lorenz_sys.plot3d(save_folder=save_folder)
    lorenz_sys.plot2d(save_folder=save_folder)
    lorenz_sys.plot_time_series(save_folder=save_folder)


