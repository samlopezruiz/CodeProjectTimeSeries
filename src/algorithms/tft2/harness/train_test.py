import datetime

import numpy as np

from algorithms.tft2.utils.data import extract_numerical_data
import algorithms.tft2.libs.utils as utils


def valid_test_model(data_formatter, model, test, valid):
    print("Computing best validation loss")
    val_loss = model.evaluate(valid)

    print("Computing test loss")
    unscaled_output_map = predict_model(data_formatter, model, test)
    q_losses = compute_q_loss(model.quantiles, unscaled_output_map)

    print("Training completed @ {}".format(datetime.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("\nNormalised Quantile Losses for Test Data: {}".format(q_losses))

    return q_losses, unscaled_output_map


def q_loss_model(data_formatter, model, test, return_output_map=False):
    unscaled_output_map = predict_model(data_formatter, model, test)
    q_losses = compute_q_loss(model.quantiles, unscaled_output_map)

    if return_output_map:
        return q_losses, unscaled_output_map
    else:
        return q_losses


def moo_q_loss_model(data_formatter, model, test, return_output_map=False, multi_processing=False,
                     eq_weights=False):
    unscaled_output_map = predict_model(data_formatter, model, test, multi_processing)
    q_losses = compute_moo_q_loss(model.quantiles, unscaled_output_map,
                                  overwrite_q=np.ones_like(model.quantiles) * 0.5 if eq_weights else None)

    if return_output_map:
        return q_losses, unscaled_output_map
    else:
        return q_losses


def compute_q_loss(quantiles, unscaled_output_map):
    targets = unscaled_output_map['targets']
    losses = {}
    for q in quantiles:
        key = 'p{}'.format(int(q * 100))
        losses[key + '_loss'] = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(unscaled_output_map[key]), q)
    q_losses = [p_loss.mean() for k, p_loss in losses.items()]
    return q_losses


def compute_moo_q_loss(quantiles, unscaled_output_map, overwrite_q=None):
    if overwrite_q is None:
        overwrite_q = quantiles

    targets = unscaled_output_map['targets']
    losses, losses_eq_weight = {}, {}
    for q, new_q in zip(quantiles, overwrite_q):
        key = 'p{}'.format(int(q * 100))
        losses[key + '_loss'] = utils.numpy_normalised_quantile_loss_moo(
            extract_numerical_data(targets), extract_numerical_data(unscaled_output_map[key]), new_q)

    #     if output_eq_loss:
    #         losses_eq_weight[key + '_loss'] = utils.numpy_normalised_quantile_loss_moo(
    #             extract_numerical_data(targets), extract_numerical_data(unscaled_output_map[key]), 0.5)
    #
    q_losses = [[obj.mean() for obj in p_loss] for k, p_loss in losses.items()]
    #
    # if output_eq_loss:
    #     q_eq_losses = [[obj.mean() for obj in p_loss] for k, p_loss in losses_eq_weight.items()]
    #
    #     return np.array(q_losses), np.array(q_eq_losses)
    #
    # else:
    return np.array(q_losses)


def predict_model(data_formatter, model, test, multi_processing=False):
    output_map = model.predict(test, return_targets=True, multi_processing=multi_processing)
    unscaled_output_map = {}
    for k, df in output_map.items():
        unscaled_output_map[k] = data_formatter.format_predictions(df)
    return unscaled_output_map
