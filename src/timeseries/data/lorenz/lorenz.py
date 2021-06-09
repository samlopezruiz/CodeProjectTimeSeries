import numpy as np
from scipy.integrate import odeint
import pandas as pd

from algorithms.dchange.func import direct_change
from timeseries.models.lorenz.functions.functions import train_test_split
from timeseries.plotly.plot import plotly_3d, plotly_phase_plots, plotly_time_series


def offset_series(series, offset=100):
    min_s = np.min(series)
    if min_s < 0:
        return np.array(series) - min_s + offset
    elif min_s < 1:
        return np.array(series) + offset
    else:
        return series


class Lorenz:

    def __init__(self, sigma=10., rho=-28., beta=8. / 3., granularity=1):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.initial_state = [0, 1, 1.05]
        start_time = 0
        end_time = 100
        self.x = self.y = self.z = [0]
        self.time_points = np.linspace(start_time, end_time, end_time * 100)
        self.df = None
        self.solved = False
        self.granularity = granularity

    def lorenz_system(self, current_state, t):
        # positions of x, y, z in space at the current time point
        x, y, z = current_state

        # define the 3 ordinary differential equations known as the lorenz equations
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z

        # return a list of the equations that describe the system
        return [dx_dt, dy_dt, dz_dt]

    def solve(self, time_points=None, positive_offset=False, noise=False, sigma=0.15, trend=False):
        if time_points is None:
            time_points = self.time_points
        else:
            self.time_points = time_points
        xyz = odeint(self.lorenz_system, self.initial_state, time_points)

        self.x, self.y, self.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        if noise:
            mu, sigma = 0, sigma
            # creating a noise with the same dimension as the dataset (2,2)
            self.x = self.x + np.random.normal(mu, sigma, self.x.shape)
            self.y = self.y + np.random.normal(mu, sigma, self.y.shape)
            self.z = self.z + np.random.normal(mu, sigma, self.z.shape)

        # extract the individual arrays of x, y, and z values from the array of arrays
        if positive_offset:
            self.x = offset_series(self.x)
            self.y = offset_series(self.y)
            self.z = offset_series(self.z)

        if trend:
            self.x = offset_series(self.x) + np.array(self.time_points)
            self.y = offset_series(self.y) + np.array(self.time_points)
            self.z = offset_series(self.z) + np.array(self.time_points)


        df = pd.DataFrame(data=np.array([self.x, self.y, self.z]).transpose(), index=self.time_points,
                          columns=['x', 'y', 'z'])
        self.df = df.iloc[::self.granularity, :]
        self.solved = True

    def plot3d(self, file_path=None, size=None, save=True, title_bool=True, label_scale=2):
        if self.solved:
            plotly_3d(self.df, title="Lorenz attractor 3D" if title_bool else None,
                      file_path=file_path, size=size, save=save, label_scale=label_scale)
        else:
            print("Solve ODE first")

    def plot2d(self, file_path=None, size=None, save=True, title_bool=True, label_scale=2):
        if self.solved:
            plotly_phase_plots(self.df, title="Lorenz attractor phase plane" if title_bool else None,
                               file_path=file_path, size=size, save=save, label_scale=label_scale)
        else:
            print("Solve ODE first")

    def plot_time_series(self, ini=None, end=None, file_path=None, size=None, save=True, markers='lines',
                         title_bool=True, label_scale=2):
        if self.solved:
            df = self.df
            df = df[df.index < end] if end is not None else df
            df = df[df.index >= ini] if ini is not None else df
            plotly_time_series(df, rows=list(range(3)), title="Lorenz Attactor Time Series" if title_bool else None,
                               file_path=file_path, size=size, save=save, markers=markers, label_scale=label_scale)
        else:
            print("Solve ODE first")

    def get_time_series(self):
        return [self.df['x'].values, self.df['y'].values, self.df['z'].values]

    def get_dataframe(self):
        if not self.solved:
            print("Solve ODE first")
        return self.df


def lorenz_system(start_time=0, end_time=100, granularity=1, positive_offset=False, noise=False, sigma=0.15, trend=False):
    t = np.arange(start_time, end_time * 100 + 1) / 100
    lorenz_sys = Lorenz(sigma=10., rho=28., beta=8. / 3., granularity=granularity)
    lorenz_sys.solve(t, positive_offset=positive_offset, noise=noise, sigma=sigma, trend=trend)
    xyz = np.array(lorenz_sys.get_time_series())
    df = lorenz_sys.get_dataframe()
    return df, xyz, list(df.index), lorenz_sys


def lorenz_wrapper(cfg):
    variate = cfg.get('variate', 'uni')
    granularity = cfg.get('granularity', 1)
    preproc = cfg.get('preprocess', False)
    positive_offset = True if preproc else cfg.get('positive_offset', False)
    noise = cfg.get('noise', False)
    sigma = cfg.get('sigma', 1)
    trend = cfg.get('trend', False)


    if variate == 'multi':
        res = multivariate_lorenz(test_perc=30, t_ini=15, granularity=granularity, end_time=130,
                                  y_col=0, positive_offset=positive_offset, noise=noise, sigma=sigma, trend=trend)
    else:
        res = univariate_lorenz(test_perc=30, t_ini=15, granularity=granularity, end_time=130,
                                positive_offset=positive_offset, noise=noise, sigma=sigma, trend=trend)


    lorenz_df, train, test, t_train, t_test = res

    return lorenz_df, train, test, t_train, t_test


def dc_lorenz_wrapper(cfg):
    variate = cfg.get('variate', 'uni')
    granularity = cfg.get('granularity', 1)
    preproc = cfg.get('preprocess', False)
    positive_offset = True if preproc else cfg.get('positive_offset', False)
    noise = cfg.get('noise', False)
    sigma = cfg.get('sigma', 1.5)
    trend = cfg.get('trend', False)
    dc_cfg = {'delta_y': (cfg.get('delta_y', 5)), 'delta_t': (cfg.get('delta_t', 0.01))}


    if variate == 'multi':
        res = multivariate_lorenz(test_perc=25, t_ini=15, granularity=granularity, end_time=1000,
                                  y_col=0, positive_offset=positive_offset, noise=noise, sigma=sigma, trend=trend)
    else:
        res = univariate_lorenz(test_perc=25, t_ini=15, granularity=granularity, end_time=1000,
                                positive_offset=positive_offset, noise=noise, sigma=sigma, trend=trend)


    lorenz_df, train, test, t_train, t_test = res
    # lorenz_df = lorenz_df[lorenz_df.index > 15]
    lorenz_dc = direct_change(lorenz_df['x'], dc_cfg)

    return lorenz_dc, train, test, t_train, t_test


def univariate_lorenz(test_perc=20, t_ini=15, granularity=1, end_time=100, positive_offset=False,
                      noise=False, sigma=1.5, trend=False):
    lorenz_df, xyz, t, _ = lorenz_system(end_time=end_time, granularity=granularity,
                                         positive_offset=positive_offset, noise=noise, sigma=sigma, trend=trend)
    x = lorenz_df['x']
    x = x[x.index > t_ini]
    data = np.array(x)
    test_size = int(len(x) * test_perc // 100)
    train, test = train_test_split(data, test_size)
    t_train, t_test = train_test_split(np.array(x.index), test_size)
    return lorenz_df, train, test, t_train, t_test


def multivariate_lorenz(test_perc=20, t_ini=15, granularity=1, end_time=100, y_col=0, positive_offset=False,
                        noise=False, sigma=1.5, trend=False):
    lorenz_df, xyz, t, _ = lorenz_system(end_time=end_time, granularity=granularity,
                                         positive_offset=positive_offset, noise=noise, sigma=sigma, trend=trend)
    df = lorenz_df[lorenz_df.index > t_ini]
    data = np.array(df)
    test_size = int(data.shape[0] * test_perc // 100)
    train, test = ([], [])
    test_col = df.columns[y_col]
    for col in list(lorenz_df.columns) + [test_col]:
        train1, test1 = train_test_split(np.array(df[col]), test_size)
        train.append(train1)
        test.append(test1)

    t_train, t_test = train_test_split(np.array(df.index), test_size)

    return lorenz_df, np.vstack(train).transpose(), np.vstack(test).transpose(), t_train, t_test


if __name__ == '__main__':
    save_folder = 'images'
    save_plots = True
    plot_titles = True
    df, xyz, t, lorenz_sys = lorenz_system(positive_offset=True, noise=True, granularity=5, sigma=1,
                                           end_time=130, trend=True)
    # lorenz_df, train, test, t_train, t_test = multivariate_lorenz(granularity=5, positive_offset=True, noise=True)


    # lorenz_sys.plot3d(file_path=[save_folder, 'lorenz-attractor-3d'], save=save_plots, title_bool=plot_titles)
    # lorenz_sys.plot2d(file_path=[save_folder, 'lorenz-attractor-phase-plane'], save=save_plots, title_bool=plot_titles)
    lorenz_sys.plot_time_series(file_path=[save_folder, 'lorenz-attractor-time-series'], save=save_plots,
                                title_bool=plot_titles)
