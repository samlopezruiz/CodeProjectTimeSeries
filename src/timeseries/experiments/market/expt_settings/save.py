import datetime
import os


def save_forecasts(config, experiment_name, results):
    save_folder = os.path.join(config.results_folder, experiment_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")[2:]
    results['targets'].to_csv(os.path.join(save_folder, 'targets' + time_stamp + '.csv'))
    results['p10_forecast'].to_csv(os.path.join(save_folder, 'p10_forecast' + time_stamp + '.csv'))
    results['p50_forecast'].to_csv(os.path.join(save_folder, 'p50_forecast' + time_stamp + '.csv'))
    results['p90_forecast'].to_csv(os.path.join(save_folder, 'p90_forecast' + time_stamp + '.csv'))