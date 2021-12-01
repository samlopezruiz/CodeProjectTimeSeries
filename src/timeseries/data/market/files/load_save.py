import os
import shutil
from copy import copy

import joblib
import pandas as pd

from timeseries.data.market.files.utils import list_files, get_path, load_market, save_df
from timeseries.data.market.files.volume import split_data_vol
from timeseries.utils.files import new_dir


def compress_txt_to_z(data_cfg, last_folder='src_folder'):
    data_cfg = copy(data_cfg)
    txt_files = list_files(data_cfg, suffix=".txt")
    z_files = list_files(data_cfg, suffix=".z")

    save_path = get_path(data_cfg, last_folder=last_folder)
    new_dir(save_path)
    for file_name in txt_files:
        new_file_name = file_name.replace("txt", "z")
        if new_file_name not in z_files:
            print("Processing file " + new_file_name)
            data_cfg['filename'] = file_name
            df, _ = load_market(data_cfg)
            joblib.dump(df, os.path.join(save_path, new_file_name))


def assert_features(df, original_features):
    new_features = list(df.columns)
    for feat in new_features:
        if feat not in original_features:
            print('{} not found in original features'.format(feat))


def merge_new_data(data_cfg, vol_cols=['vol', 'volp']):
    new_df_vol = None
    df_original, features = load_market(data_cfg, last_folder='dump_folder')
    z_files = list_files(data_cfg, suffix=".z", last_folder='src_folder')
    path = get_path(data_cfg, last_folder='src_folder')

    dfs = [joblib.load(os.path.join(path, z_file_name)) for z_file_name in z_files]
    df = df_original
    for new_df, z_file_name in zip(dfs, z_files):
        if vol_cols[0] in new_df.columns:
            new_df, new_df_vol = split_data_vol(new_df, vol_cols=vol_cols)
        assert_features(new_df, features)
        ini_ix = df_original.index[-1]
        df_ss = new_df.loc[ini_ix:, :]
        df = pd.concat([df, df_ss], axis=0)

        if new_df_vol is not None:
            vol_path = get_path(data_cfg, last_folder='src_folder')
            joblib.dump(new_df_vol, os.path.join(vol_path, z_file_name.replace(".z", "")+"_vol.z"))

    save_df(df, data_cfg, last_folder='dump_folder')
    return df_original, df

def move_used_files(data_cfg, last_folder='raw_folder', exclude_substring='', del_all=False):
    txt_files = list_files(data_cfg, suffix=".txt")
    z_files = list_files(data_cfg, suffix=".z")
    path = get_path(data_cfg)
    new_path = get_path(data_cfg, last_folder=last_folder)
    new_dir(new_path)
    for z_file in z_files:
        if exclude_substring not in z_file or len(exclude_substring) == 0:
            if del_all:
                os.remove(os.path.join(path, z_file))
            else:
                shutil.move(os.path.join(path, z_file), new_path)
    for txt_file in txt_files:
        os.remove(os.path.join(path, txt_file))
