import logging
import numpy as np

def get_model_func_names(model_func):
    new_dict = {}
    for item in model_func:
        if hasattr(model_func[item], '__name__'):
            new_dict[item] = model_func[item].__name__
    return new_dict

def set_logging():
    logging.basicConfig(filename='results.log',
                        level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')
    return logging


def log_forecast_results(data_mkt_cfg, data_reg_cfg, training_cfg, model_cfg, results,
                         all_forecast_df, cm_metrics, model_func, train_features):
    set_logging()
    logging.info('---NEW RUN ---')
    logging.info('Market Data: {}'.format(data_mkt_cfg['filename']))
    logging.info('Regime Data: {}'.format(data_reg_cfg['filename']))
    logging.info('Training Cfg: {}'.format(str(training_cfg)))
    logging.info('Model Cfg: {}'.format(str(model_cfg)))
    logging.info('Model Func: {}'.format(str(get_model_func_names(model_func))))
    logging.info('Train Features: {}'.format(str(train_features)))
    logging.info('No. Test Subsets: {}'.format(len(results)))
    logging.info('Test Shape: {}'.format(all_forecast_df.shape[0]))
    for m in results.columns:
        logging.info('Test subsets {}: {} +-({})'.format(m, round(np.mean(results[m]), 2),
                                                         round(np.std(results[m]), 4)))
    logging.info('Test rsme: {} +-({})'.format(round(np.mean(all_forecast_df['rse']), 2),
                                               round(np.std(all_forecast_df['rse']), 2)))
    logging.info('Hit Rate: {} %'.format(round(100 * sum(all_forecast_df['hit_rate']) / all_forecast_df.shape[0], 4)))
    logging.info('Hits Confusion Matrix: {}'.format(str(cm_metrics)))
