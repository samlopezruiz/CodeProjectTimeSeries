from timeseries.experiments.market.utils.filename import hmm_filename, subsets_and_test_filename, subset_filename
from timeseries.experiments.utils.files import save_vars


def save_hmm(var, in_cfg, hmm_cfg, use_date_suffix=False):
    file_name = hmm_filename(hmm_cfg)
    if in_cfg['save_results']:
        save_vars(var, [in_cfg['results_folder'], file_name], use_date_suffix=use_date_suffix)


def save_subsets_and_test(var, in_cfg, data_cfg, split_cfg, use_date_suffix=False):
    file_name = subsets_and_test_filename(data_cfg, split_cfg)
    if in_cfg['save_results']:
        save_vars(var, [in_cfg['results_folder'], file_name], use_date_suffix=use_date_suffix)


def save_market_data(var, in_cfg, data_cfg, use_date_suffix=False):
    file_name = subset_filename(data_cfg)
    if in_cfg['save_results']:
        save_vars(var, [in_cfg['results_folder'], file_name], use_date_suffix=use_date_suffix)


