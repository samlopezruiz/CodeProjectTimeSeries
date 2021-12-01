from timeseries.data.market.files.load_save import compress_txt_to_z, merge_new_data, move_used_files
from timeseries.plotly.plot import plotly_merge

if __name__ == '__main__':
    data_cfg = {'inst': "NQ", 'suffix': "2012-2020", 'sampling': 'day',
                'src_folder': "new", 'dump_folder': "data", 'market': 'cme'}

    compress_txt_to_z(data_cfg)
    df_original, df = merge_new_data(data_cfg)
    plotly_merge(df_original, df, data_cfg['inst'], ix_range=50)
    move_used_files(data_cfg)



