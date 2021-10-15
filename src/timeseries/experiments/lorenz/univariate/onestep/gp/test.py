from deap import gp
import matplotlib.pyplot as plt
from collections import deque
from timeseries.data.lorenz.lorenz import univariate_lorenz, lorenz_wrapper
from timeseries.experiments.lorenz.functions.dataprep import split_uv_seq_multi_step
from timeseries.experiments.lorenz.univariate.onestep.gp.func import train_gp
from timeseries.experiments.lorenz.univariate.onestep.gp.plotgp import plot_gp_log, plot_ind, hierarchy_pos
from timeseries.experiments.utils.forecast import multi_step_forecast_df
from timeseries.experiments.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series


if __name__ == "__main__":
    in_steps = 10
    out_steps = 1
    pred_steps = 1
    ngen = 60
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True}
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    name = "GP"
    cfg = {"in_steps": in_steps, "ngen": ngen}

    # %%
    toolbox, pset, pop, log, hof = train_gp(train, in_steps=in_steps,
                                            out_steps=out_steps, ngen=ngen)
    plot_gp_log(log)

    pop_wo_nans = [p for p in pop if p.fitness.values[0] > 0]
    pop_wo_nans.sort(key=lambda x: x.fitness, reverse=True)
    for i in range(10):
        print("{}: {} = {}".format(i, str(pop_wo_nans[i]), round(toolbox.evaluate(pop_wo_nans[i])[0], 6)))

    # %%
    best = pop_wo_nans[0]
    nodes, edges, labels = gp.graph(best)
    plot_ind(best, root=0)

    # %%
    ts = test
    X, Y = split_uv_seq_multi_step(ts, in_steps, out_steps)
    func = gp.compile(best, pset)
    y_pred = list(X[0])
    n = in_steps
    errs = deque(maxlen=n)
    for _ in range(n):
        errs.append(0)
    for x, Y_obs in zip(X, Y):
        history = list(x)
        for i, y in enumerate(Y_obs):
            inputs = list(errs) + history[-n:]
            y_p = func(*inputs)
            y_pred.append(func(*inputs))
            if i == 0:
                errs.append(y - y_p)
            history.append(y_p)

    forecast = y_pred
    df = multi_step_forecast_df(train, test, forecast, train_prev_steps=500)
    plotly_time_series(df, title=name+" | "+str(input_cfg)+" | "+str(cfg), markers='lines')
    metrics = forecast_accuracy(forecast, test)
    print(metrics)














