from timeseries.models.market.utils.filename import hmm_filename, subsets_and_test_filename, subset_filename
from timeseries.models.utils.models import save_vars


def save_hmm(var, in_cfg, hmm_cfg):
    file_name = hmm_filename(hmm_cfg)
    save_vars(var, [in_cfg['results_folder'], file_name], in_cfg['save_results'])


def save_subsets_and_test(var, in_cfg, data_cfg, split_cfg):
    file_name = subsets_and_test_filename(data_cfg, split_cfg)
    save_vars(var, [in_cfg['results_folder'], file_name], in_cfg['save_results'])

def save_market_data(var, in_cfg, data_cfg):
    file_name = subset_filename(data_cfg)
    save_vars(var, [in_cfg['results_folder'], file_name], in_cfg['save_results'])
