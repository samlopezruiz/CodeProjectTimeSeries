import numpy as np


def test_construction(lorenz_sys):
    assert lorenz_sys.sigma == 10
    assert lorenz_sys.rho == 28
    assert lorenz_sys.beta == 8. / 3.
    assert lorenz_sys.initial_state == [0.1, 0, 0]

def test_solution(lorenz_sys):
    start_time = 0
    end_time = 100
    time_points = np.linspace(start_time, end_time, end_time * 100)
    x, y, z = lorenz_sys.solve(time_points)
    assert len(x) == len(time_points)
    assert len(y) == len(time_points)
    assert len(z) == len(time_points)