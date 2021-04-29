import numpy as np
from numpy import mean, std


def consolidate_summary(results, names, score_type):
    cfg_scores, cfg_times, cfg_params, cfg_loss = [], [], [], []
    summary = []
    for i, res in enumerate(results):
        metrics, _, times, n_params, loss = res
        scores, scores_m, score_std = summarize_scores(names[i], metrics, score_type)
        train_t_m, train_t_std, pred_t_m, pred_t_std = summarize_times(names[i], times)
        # cfg_times.append((names[i], times, train_t_m, train_t_std, pred_t_m, pred_t_std))
        # cfg_scores.append((names[i], scores, scores_m, score_std))
        # cfg_params.append((names[i], n_params))
        # cfg_loss.append((names[i], np.mean(loss), np.std(loss)))
        summary.append((names[i], scores, scores_m, score_std, np.mean(loss), np.std(loss), times, train_t_m,
                        train_t_std, pred_t_m, pred_t_std, n_params))

    return summary


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