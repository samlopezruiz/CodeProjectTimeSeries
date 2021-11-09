import pandas as pd
import os
from datetime import date
import datetime

from timeseries.experiments.utils.files import create_dir, get_new_file_path
from timeseries.utils.files import new_dir

template = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "none"]


def plotly_params_check(df, instrument=None, **kwargs):
    if not isinstance(df, pd.DataFrame):
        print("ERROR: First parameter is not pd.DataFrame")
        return False, None

    params_ok = True
    features = kwargs.get('features') if kwargs.get('features', None) is not None else df.columns
    f = len(features)
    rows = kwargs.get('rows') if kwargs.get('rows', None) is not None else [0 for _ in range(f)]
    cols = kwargs.get('cols') if kwargs.get('cols', None) is not None else [0 for _ in range(f)]
    alphas = kwargs.get('alphas') if kwargs.get('alphas', None) is not None else [1 for _ in range(f)]
    type_plot = kwargs.get('type_plot') if kwargs.get('type_plot', None) is not None else ["line" for _ in range(f)]

    for feature in features:
        if feature not in df.columns:
            print("ERROR: feature ", feature, "not found")
            params_ok = False

    if len(rows) != f:
        print("ERROR: len(rows) != features")
        params_ok = False

    if len(cols) != f:
        print("ERROR: len(cols) != features")
        params_ok = False

    if len(type_plot) != f:
        print("ERROR: len(type_plot) != features")
        params_ok = False

    return params_ok, (list(features), rows, cols, type_plot, alphas)


def set_y_labels(f, features, fig):
    for i in range(f):
        fig['layout']['yaxis' + str(i + 1)]['title'] = features[i]


def plotly_save(fig, file_path, size, save_png=False, use_date_suffix=False):
    print("saving .html and/or .png")
    create_dir(file_path)
    image_path = get_new_file_path(file_path, '.png', use_date_suffix)
    html_path = get_new_file_path(file_path, '.html', use_date_suffix)
    if size is None:
        size = (1980, 1080)

    if save_png:
        fig.write_image(os.path.join(image_path), width=size[0], height=size[1], engine='orca')

    fig.write_html(os.path.join(html_path))


