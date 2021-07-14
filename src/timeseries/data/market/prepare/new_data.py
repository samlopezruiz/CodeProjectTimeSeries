from timeseries.data.market.files.load_save import merge_new_data, move_used_files, compress_txt_to_z
from timeseries.data.market.files.volume import calc_cumm_vp, merge_vol
from timeseries.plotly.plot import plotly_merge

if __name__ == '__main__':
    data_cfg = {'inst': "NQ", 'suffix': "2012-2020", 'sampling': 'minute', 'market': 'cme',
                'src_folder': "new", 'raw_folder': 'raw', 'dump_folder': "data", 'split_folder': 'split',
                'vol_folder': 'vol', 'vol_suffix': '2012-2020_vol'
                }
    # Put in 'src_folder' the new txt file
    # 'suffix' is the name of the compiled file so far (in dump_folder)
    save_vol = False

    compress_txt_to_z(data_cfg)
    df_original, df = merge_new_data(data_cfg)
    plotly_merge(df_original, df, data_cfg['inst'], ix_range=50)
    move_used_files(data_cfg, exclude_substring='vol')
    if save_vol:
        vp_cumm = calc_cumm_vp(data_cfg)
        vp_complete = merge_vol(data_cfg, vp_cumm)
    else:
        vp_complete = None
    if vp_complete is not None or not save_vol:
        move_used_files(data_cfg, del_all=True)


