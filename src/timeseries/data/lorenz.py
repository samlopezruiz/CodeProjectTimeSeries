import numpy as np
from scipy.integrate import odeint


class Lorenz:

    def __init__(self, sigma=10., rho=28., beta=8. / 3.):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.initial_state = [0.1, 0, 0]
        start_time = 0
        end_time = 100
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
        xyz = odeint(self.lorenz_system, self.initial_state, time_points)

        # extract the individual arrays of x, y, and z values from the array of arrays
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        return x, y, z
