import numpy as np
from numpy import mean, std


def consolidate_summary(results, names, score_type):
    summary = []
    for i, res in enumerate(results):
        metrics, _, times, n_params, loss = res
        scores, scores_m, score_std = summarize_scores(names[i], metrics, score_type)
        train_t_m, train_t_std, pred_t_m, pred_t_std = summarize_times(names[i], times)
        summary.append((names[i], scores, scores_m, score_std, np.mean(loss), np.std(loss), times, train_t_m,
                        train_t_std, pred_t_m, pred_t_std, n_params))

    return summary


def consolidate_series_summaries(results, names, score_type):
    model = {}
    for name in names:
        model[name] = {}

    for metric in ['score', 'train_t', 'pred_t', 'loss', 'params']:
        for name in names:
            model[name][metric] = []

    for result in results:
        for i, res in enumerate(result):
            score_types, _, times, n_params, loss = res
            score = [m[score_type] for m in score_types]
            times = np.array(times)
            train_t, pred_t = times[:, 0], times[:, 1]
            model[names[i]]['score'] += list(score)
            model[names[i]]['train_t'] += list(train_t)
            model[names[i]]['pred_t'] += list(pred_t)
            model[names[i]]['loss'] += list(loss)
            model[names[i]]['params'] = n_params

    consolidate = []
    for name, metric in model.items():
        consolidate.append((name, metric['score'], np.mean(metric['score']), np.std(metric['score']),
                            np.mean(metric['loss']), np.std(metric['loss']), None, np.mean(metric['train_t']),
                            np.std(metric['train_t']), np.mean(metric['pred_t']),
                            np.std(metric['pred_t']), metric['params']))

    return consolidate


def summarize_scores(name, metrics, score_type='rmse'):
    scores = [m[score_type] for m in metrics]
    scores_m, score_std = mean(scores), std(scores)
    print('{}: {} {}  (+/- {})'.format(name, round(scores_m, 4), score_type, round(score_std, 4)))
    return scores, scores_m, score_std


def summarize_times(name, times):
    times = np.array(times)
    train_t, pred_t = times[:, 0], times[:, 1]
    train_t_m, train_t_std = mean(train_t), std(train_t)
    pred_t_m, pred_t_std = mean(pred_t), std(pred_t)
    print('{}: train = {} s  (+/- {}), pred = {} s  (+/- {})'.format(name, round(train_t_m, 1), round(train_t_std, 4),
                                                                     round(pred_t_m, 4), round(pred_t_std, 4)))
    return train_t_m, train_t_std, pred_t_m, pred_t_std