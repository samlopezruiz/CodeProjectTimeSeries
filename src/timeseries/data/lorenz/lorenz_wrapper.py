import numpy as np
from timeseries.data.lorenz.lorenz import Lorenz


def lorenz_system(start_time=0, end_time=100):
    t = np.arange(start_time, end_time * 100 + 1) / 100
    lorenz_sys = Lorenz(sigma=10., rho=28., beta=8. / 3.)
    lorenz_sys.solve(t)
    xyz = lorenz_sys.get_time_series()
    df = lorenz_sys.get_dataframe()
    return df, xyz, t
