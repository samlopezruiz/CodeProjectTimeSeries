def models_strings(names, model_cfgs):
    models_info, models_name = '', ''
    for i, name in enumerate(names):
        models_info += '<br>' + name + ': ' + str(model_cfgs[i])
        models_name += name + '_'

    return models_info, models_name