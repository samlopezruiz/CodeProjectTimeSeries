

def hmm_filename(hmm_cfg):
    return ('regime_' + '_'.join(hmm_cfg['hmm_vars'])).replace("^", "")


def subsets_and_test_filename(data_cfg, split_cfg):
    return 'split_' + data_cfg['inst'] + '_' + data_cfg['sampling'] + '_' + data_cfg['data_from'] + '_to_' + \
            data_cfg['data_to'] + '_g' + str(split_cfg['groups_of']) + split_cfg['group'] + \
            '_r' + str(split_cfg['test_ratio'])
