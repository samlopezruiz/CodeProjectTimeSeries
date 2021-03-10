import pandas as pd
import os

template = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "none"]


def plotly_params_check(df, **kwargs):
    if not isinstance(df, pd.DataFrame):
        print("ERROR: First parameter is not pd.DataFrame")
        return False, None

    params_ok = True
    features = kwargs.get('features') if kwargs.get('features', None) is not None else df.columns
    f = len(features)
    rows = kwargs.get('rows') if kwargs.get('rows', None) is not None else [0 for _ in range(f)]
    cols = kwargs.get('cols') if kwargs.get('cols', None) is not None else [0 for _ in range(f)]
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

    return params_ok, (features, rows, cols, type_plot)

def set_y_labels(f, features, fig):
    for i in range(f):
        fig['layout']['yaxis' + str(i + 1)]['title'] = features[i]


def plotly_save(fig, file_path, size):
    print("saving .html and .png")
    if not os.path.exists(file_path[0]):
        os.makedirs(file_path[0])
    image_path = file_path[:-1].copy() + [file_path[-1] + ".png"]
    html_path = file_path[:-1].copy() + [file_path[-1] + ".html"]
    if size is None:
        size = (1980, 1080)
    fig.write_image(os.path.join(*image_path), width=size[0], height=size[1])
    fig.write_html(os.path.join(*html_path))