import numpy as np


def test_construction(lorenz_sys):
    assert lorenz_sys.sigma == 10
    assert lorenz_sys.rho == 28
    assert lorenz_sys.beta == 8. / 3.
    assert lorenz_sys.initial_state == [0, 1, 1.05]


def test_solution(lorenz_sys):
    start_time = 0
    end_time = 100
    time_points = np.linspace(start_time, end_time, end_time * 100)
    lorenz_sys.solve(time_points)
    xyz = lorenz_sys.get_time_series()
    assert len(xyz[0]) == len(time_points)
    assert len(xyz[1]) == len(time_points)
    assert len(xyz[2]) == len(time_points)
