import os
import pandas as pd
import numpy as np
import joblib
import copy

from timeseries.data.market.files.utils import find_filenames, DATA_ROOT
from timeseries.experiments.market.volume.utils import split_cumm_vol

if __name__ == '__main__':
    data_cfg = {
        'contract': "03-21",
        'sample': 'minute',
        'inst': "NQ",
        'src_folder': "raw",
        'new_folder': "split"
    }

    save_vol = False

    path = os.path.join(DATA_ROOT, 'cme', data_cfg['sample'], data_cfg['inst'])
    z_files = find_filenames(os.path.join(path, data_cfg['src_folder']), suffix=".z")
    z_files.sort()

    for i in range(0, len(z_files)):
        if i == 0:
            split_cumm_vol(path,
                           data_cfg['src_folder'],
                           data_cfg['new_folder'],
                           z_files[i],
                           save_volume=save_vol)
        else:
            split_cumm_vol(path,
                           data_cfg['src_folder'],
                           data_cfg['new_folder'],
                           z_files[i],
                           z_files[i - 1],
                           save_volume=save_vol)
