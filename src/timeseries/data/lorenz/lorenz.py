import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

from timeseries.plot.plot_utils import plotly_row_traces, plotly_save, plotly_phase_plots, plotly_3d
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from itertools import combinations


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

    def plot3d(self, file_path=None, size=None):
        if len(self.x) > 1:
            df = pd.DataFrame(data=np.array([self.x, self.y, self.z]).transpose(), index=self.time_points,
                              columns=['x', 'y', 'z'])
            plotly_3d(df, title="Lorenz attractor 3D",
                               file_path=file_path, size=size)
        else:
            print("Solve ODE first")

    def plot2d(self, file_path=None, size=None):
        if len(self.x) > 1:
            df = pd.DataFrame(data=np.array([self.x, self.y, self.z]).transpose(), index=self.time_points,
                              columns=['x', 'y', 'z'])
            plotly_phase_plots(df, title="Lorenz attractor phase plane",
                               file_path=file_path, size=size)
        else:
            print("Solve ODE first")

    def plot_time_series(self, file_path=None, size=None):
        if len(self.x) > 1:
            df = pd.DataFrame(data=np.array([self.x, self.y, self.z]).transpose(), index=self.time_points,
                              columns=['x', 'y', 'z'])
            plotly_row_traces(df, title="Lorenz Attactor Time Series",
                              file_path=file_path, size=size)
        else:
            print("Solve ODE first")

    def get_time_series(self):
        return [self.x, self.y, self.z]


if __name__ == '__main__':
    save_folder = 'images'
    start_time = 0
    end_time = 100
    t = np.linspace(start_time, end_time, end_time * 100 + 1, dtype='float64')
    # t = np.arange(start_time, end_time * 100 + 1) / 100

    lorenz_sys = Lorenz(sigma=10., rho=28., beta=8. / 3.)
    lorenz_sys.solve(t)
    xyz = lorenz_sys.get_time_series()
    lorenz_sys.plot3d(file_path=[save_folder, 'lorenz-attractor-3d'])
    lorenz_sys.plot2d(file_path=[save_folder, 'lorenz-attractor-phase-plane'])
    lorenz_sys.plot_time_series(file_path=[save_folder, 'lorenz-attractor-time-series'])
