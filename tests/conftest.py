import pytest

from timeseries.data.lorenz import Lorenz


@pytest.fixture
def lorenz_sys() -> Lorenz:
    return Lorenz(sigma=10., rho=28., beta=8. / 3.)

