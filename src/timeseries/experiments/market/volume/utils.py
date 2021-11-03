import copy
import os

import joblib

from timeseries.data.market.files.utils import find_filenames, load_data


def split_cumm_vol(path, src_folder, new_folder, filename1, filename0=None, save_volume=True):
    df1, vol1 = split_data_vol(os.path.join(path, src_folder), filename1, save_volume)
    csv_data_name1, csv_vol_name1 = get_vol_data_filenames(filename1)
    save_data(df1, path, new_folder, csv_data_name1)
    if vol1 is not None and save_volume:
        save_vol(vol1, path, new_folder, csv_vol_name1, filename0)


def last_vol_prof(path, new_folder, filename0, split_z_files):
    if filename0 is not None:
        vol_filename0 = filename0.replace(".z", " vol.z")
        vol0_path = os.path.join(path, new_folder, vol_filename0)
        if vol_filename0 in split_z_files:
            vp_cumm0 = joblib.load(vol0_path)
            _, volume_profile = vp_cumm0[-1]
            return volume_profile
        else:
            print("File {} not found".format(vol_filename0))
            return {}
    else:
        return {}


def get_sessions_ix(dt, rth_start=8.5, rth_end=15.25):
    session_ends = []
    for i in range(1, len(dt)):
        t1 = dt[i - 1].hour + dt[i - 1].minute / 60
        t0 = dt[i].hour + dt[i].minute / 60
        if t1 < rth_start and t0 >= rth_start:
            session_ends.append(i - 1)
        elif t1 < rth_end and t0 >= rth_end:
            session_ends.append(i - 1)
    return session_ends


def get_cumm_vp(vol1, session_ends, volume_profile, dt):
    vp_cumm = []
    for sess in session_ends:
        vol = vol1.iloc[sess, 0]
        volp = vol1.iloc[sess, 1]
        for v, price in enumerate(volp):
            if price not in volume_profile:
                volume_profile[price] = vol[v]
            else:
                volume_profile[price] += vol[v]
        vp_cumm.append((dt[sess], copy.copy(volume_profile)))
    return vp_cumm


def get_last_vol_prof(path, new_folder, filename0):
    split_z_files = find_filenames(os.path.join(path, new_folder), suffix=".z")
    return last_vol_prof(path, new_folder, filename0, split_z_files)


def get_vol_data_filenames(filename1):
    csvname1 = filename1.replace(".z", "")
    return csvname1 + ".csv", csvname1 + " vol.z"


def split_data_vol(path, filename1, save_vol):
    df1, _ = load_data(path, filename=filename1)
    vol_cols = ['vol', 'volp']
    if vol_cols[0] in df1.columns and save_vol:
        vol1 = df1[vol_cols]
        df1.drop(vol_cols, axis=1, inplace=True)
        return df1, vol1
    else:
        df1.drop(vol_cols, axis=1, inplace=True)
        return df1, None


def save_data(df1, path, new_folder, csv_data_name1):
    df1.to_csv(os.path.join(path, new_folder, csv_data_name1), index=True)
    print("File {} saved".format(csv_data_name1))


def save_vol(vol1, path, new_folder, csv_vol_name1, filename0):
    dt = list(vol1.index)
    volume_profile = get_last_vol_prof(path, new_folder, filename0)
    session_ends = get_sessions_ix(dt)
    vp_cumm = get_cumm_vp(vol1, session_ends, volume_profile, dt)
    joblib.dump(vp_cumm, os.path.join(path, new_folder, csv_vol_name1))
    print("File {} saved".format(csv_vol_name1))
