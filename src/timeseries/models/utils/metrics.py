import numpy as np
from statsmodels.tsa.stattools import acf
import pandas as pd
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast


def forecast_accuracy(forecast, test):
    omin = -np.min(np.hstack([forecast, test]))
    forecast = np.array(forecast) + omin + 1
    test = np.array(test) + omin + 1
    mape = np.mean(np.abs(forecast - test) / np.abs(test))  # MAPE
    me = np.mean(forecast - test)  # ME
    mae = np.mean(np.abs(forecast - test))  # MAE
    mpe = np.mean((forecast - test) / test)  # MPE
    rmse = np.mean((forecast - test) ** 2) ** .5  # RMSE
    corr = np.corrcoef(forecast, test)[0, 1]  # corr
    mins = np.amin(np.vstack([forecast, test]), axis=0)
    maxs = np.amax(np.vstack([forecast, test]), axis=0)
    minmax = np.mean(mins / maxs)
    # acf1 = acf(forecast - test, fft=False)[1]  # ACF1
    return ({'mape': round(mape, 4), 'me': round(me, 4), 'mae': round(mae, 4),
             'mpe': round(mpe, 4), 'rmse': round(rmse, 4),
             'corr': round(corr, 4), 'minmax': round(minmax, 4)}) #'acf1': round(acf1, 4),


def compare_forecast(models, train, test):
    metrics = []
    for model_name, (creation, forecast, cfg) in models.items():
        print("Forecast with {}".format(model_name))
        error, forecast = walk_forward_step_forecast(train, test, cfg, forecast, creation)
        metrics.append(forecast_accuracy(forecast, test))

    df = pd.DataFrame(metrics).transpose()
    df.columns = models.keys()
    return df


def summary_results(consolidated, score_type='score', less_is_better=False):
    df = pd.DataFrame(consolidated, columns=['model', 'scores', 'score_m', 'score_std','loss_m', 'loss_std',
                                             'times', 'train_t_m', 'train_t_std', 'pred_t_m', 'pred_t_std',  'n_params'])
    df.set_index(['model'], inplace=True)
    df.drop(['times', 'scores'], axis=1, inplace=True)
    normalized_df = (df - df.min()) / (df.max() - df.min())
    normalized_df = normalized_df.fillna(1)
    normalized_df['train_t_m'] = normalized_df['train_t_m'].max() - normalized_df['train_t_m']
    normalized_df['pred_t_m'] = normalized_df['pred_t_m'].max() - normalized_df['pred_t_m']
    normalized_df['n_params'] = normalized_df['n_params'].max() - normalized_df['n_params']
    if less_is_better:
        normalized_df['score_m'] = normalized_df['score_m'].max() - normalized_df['score_m']

    df['overall'] = 3 * normalized_df['score_m'] + normalized_df['train_t_m'] + \
                    1.5 * normalized_df['pred_t_m'] + normalized_df['n_params']
    df = df.sort_values(by=['overall'], ascending=False)
    data, errors = get_data_error(df, score_type)
    return df, data, errors

def get_data_error(summary, score_type):
    data = summary.copy()
    errors = summary.loc[:, ['score_std', 'loss_std', 'train_t_std', 'pred_t_std']]
    data.drop(['score_std', 'loss_std', 'train_t_std', 'pred_t_std'], axis=1, inplace=True)
    data.columns = [score_type, 'train mse', 'train (s)', 'pred (s)', 'n params', 'overall score']
    return data, errors