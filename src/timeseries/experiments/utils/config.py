def unpack_input_cfg(cfg):
    variate, granularity, noise = cfg['variate'], cfg['granularity'], cfg['noise']
    trend, detrend, preprocess = cfg['trend'], cfg['detrend'], cfg['preprocess']
    return variate, granularity, noise, trend, detrend, preprocess


def unpack_in_cfg(in_cfg):
    image_folder, results_folder = in_cfg['image_folder'], in_cfg['results_folder']
    save_results, plot_hist, verbose = in_cfg['save_results'], in_cfg.get('plot_hist', False), in_cfg.get('verbose', 0)
    plot_title = in_cfg['plot_title']
    return image_folder, plot_hist, plot_title, save_results, results_folder, verbose