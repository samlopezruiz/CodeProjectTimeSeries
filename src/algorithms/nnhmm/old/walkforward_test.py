# X_test, y_test, X_reg_test = model_func['xy_from_train'](test_pp, *model_func['xy_args'](model_cfg, test_reg_prob))
    # X_bundle = X_test[::pred_steps, :]
    # reg_bundle = np.zeros((X_bundle.shape[0], 1)) if X_reg_test is None else X_reg_test[::pred_steps, :]
    #
    # # y_bundle = y[::pred_steps, :]
    # y_test_bundles = y_test[::model_n_steps_out, :] #.ravel()
    # %%
    # history, test_bundles, y_test = get_bundles(True, model_n_steps_out, test_pp, train_pp)
    # %%
    # y_pred = []
    # for i, (x, reg_prob) in enumerate(zip(X_bundle, reg_bundle)):
    #     # expand dims to indicate just one prediction
    #     if use_regimes:
    #         model_input_test = [np.expand_dims(x, axis=0), np.expand_dims(reg_prob, axis=0)]
    #     else:
    #         model_input_test = np.expand_dims(x, axis=0)
    #     y_hat = model.predict(model_input_test)[0][:pred_steps]
    #     y_pred.append(y_hat)
    #
    # y_pred = np.hstack(y_pred)
    # y_forecast = y_pred.ravel()
    # y_true, y_forecast = trim_min_len(y_true, y_forecast)
    #
    # # %%
    # # test data includes first value and has len greater by 1
    # test_y = test_x[:, -1][:len(y_forecast)]
    #
    # # %%