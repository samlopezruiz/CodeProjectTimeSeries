import datetime
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.preprocessing import preprocess
from timeseries.utils.files import new_dir

# sns.set_theme(style="white")
sns.set_theme()
sns.set_context("poster", font_scale=2)

if __name__ == '__main__':
    # %% GENERAL INPUTS
    save = True

    input_cfg = {"variate": "multi", "granularity": 5, "noise": False, 'preprocess': False,
                 'trend': False, 'detrend': 'ln_return'}

    # %% DATA
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    df = pd.DataFrame(train_pp[:, :3], columns=['x', 'y', 'z'])
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, ax=ax)
    if save:
        new_dir('images')
        file_name = 'corr' + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H-%M") + ".png"
        plt.savefig(os.path.join('images', file_name))

    plt.show()