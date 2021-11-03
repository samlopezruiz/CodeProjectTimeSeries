import os
import pandas as pd
from _utils import find_filenames
import joblib

#%% CONSTANTS
contract = "03-21"
ROOT = "../historical_data/cme"
sample = 'minute'
inst = "ES"
src_folder = "split"
new_folder = "vol"
new_filename = "2012-2020 vol"
src_path = os.path.join(ROOT,sample,inst,src_folder)
new_path = os.path.join(ROOT,sample,inst,new_folder)

z_files = find_filenames(src_path, suffix=".z" )
z_files.sort()

#%%

vp_df = []
for z_file in z_files:
    print("File {} is being processed".format(z_file))
    volume_profile = joblib.load(os.path.join(src_path,z_file))
    dfs = []
    # for ix, vp in volume_profile:
    #     dfs.append(pd.DataFrame(vp,index=[ix]).astype(int))
    
    vp_df.append(pd.concat(dfs))

#%%
result = pd.concat(vp_df)
sorted_prices = list(result.columns)
result = result[sorted(sorted_prices)]
result.to_csv(os.path.join(new_path,inst+" "+new_filename+".csv"), index=True)
