import os
import pandas as pd

from timeseries.data.market.files.utils import find_filenames


def merge_files(src_path, new_path, filename):
    csv_files = find_filenames(src_path, suffix=".csv")
    csv_files.sort()

    result = concat_dfs(src_path, csv_files)

    print("New File Shape: {}".format(result.shape))
    new_filename = inst + " " + filename + ".csv"
    result.to_csv(os.path.join(new_path, new_filename), index=True)
    print("File {} saved".format(new_filename))


def concat_dfs(src_path, csv_files):
    dfs = [pd.read_csv(os.path.join(src_path, csv_file),
                       infer_datetime_format=True,
                       parse_dates=["datetime"], index_col=["datetime"])
           for csv_file in csv_files]

    return pd.concat(dfs)


# %% CONSTANTS
contract = "03-21"
ROOT = "../historical_data/compressed"
sample = 'tick'
inst = "ES"
src_folder = "split"
new_folder = "data"
new_filename = "2012-2020"
src_path = os.path.join(ROOT, sample, inst, src_folder)
new_path = os.path.join(ROOT, sample, inst, new_folder)

merge_files(src_path, new_path, new_filename)
