def get_params(model, model_cfg):
    if isinstance(model_cfg, list):
        params = 0
        for i, cfg in enumerate(model_cfg):
            name, input_cfg, model_cfg, func_cfg = cfg
            depth = model_cfg.get('depth', None)
            if depth is None:
                params += model[i].count_params()
            else:
                params += 2 ** (cfg.get('depth', 1) - 2) * 6
    else:
        params = 0
        depth = model_cfg.get('depth', None)
        if depth is None:
            if hasattr(model, 'count_params'):
                params = model.count_params()
            if hasattr(model, 'param_names'):
                params = len(model.param_names)
        else:
            params = 2 ** (model_cfg.get('depth', 1) - 2) * 6
    return params