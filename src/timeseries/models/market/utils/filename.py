
def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def hmm_filename(hmm_cfg):
    return ('regime_' + '_'.join(hmm_cfg['hmm_vars'])).replace("^", "")

def res_mkt_filename(data_cfg, training_cfg, model_cfg):
    data_name = data_cfg['filename'][find_nth(data_cfg['filename'], '_', 1):find_nth(data_cfg['filename'], '_', 3)]
    return 'res' + data_name + '_y' + training_cfg['y_true_var'] + '_m' + model_cfg['name'] + '_reg' + str(model_cfg['use_regimes'])

def subsets_and_test_filename(data_cfg, split_cfg):
    d_text = dwn_smple_text(data_cfg)
    return 'split_' + data_cfg['inst'] + '_' + data_cfg['sampling'] + d_text + '_' + data_cfg['data_from'] + '_to_' + \
            data_cfg['data_to'] + '_g' + str(split_cfg['groups_of']) + split_cfg['group'] + \
            '_r' + str(split_cfg['test_ratio'])[-2:]

def subset_filename(data_cfg):
    d_text = dwn_smple_text(data_cfg)
    return 'subset_' + data_cfg['inst'] + '_' + data_cfg['sampling'] + d_text + '_' + data_cfg['data_from'] + '_to_' + \
            data_cfg['data_to']


def dwn_smple_text(data_cfg):
    if data_cfg.get('downsample', False):
        d_text = '_' + data_cfg['downsample_p'] + '_dwn_smpl'
    else:
        d_text = ''
    return d_text

# def subsets_and_test_filename(data_cfg, split_cfg):
#     return 'split_' + data_cfg['inst'] + '_' + data_cfg['sampling'] + '_' + data_cfg['data_from'] + '_to_' + \
#             data_cfg['data_to'] + '_g' + str(split_cfg['groups_of']) + split_cfg['group'] + \
#             '_r' + str(split_cfg['test_ratio'])
