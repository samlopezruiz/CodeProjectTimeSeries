from timeseries.data.lorenz.lorenz import lorenz_system
from pandas_profiling import ProfileReport


if __name__ == '__main__':
    save_folder = 'images'
    save_plots = False
    df, _, _, lorenz_sys = lorenz_system()
    df.reset_index(drop=True, inplace=True)
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file("Lorenz Profiling.html")