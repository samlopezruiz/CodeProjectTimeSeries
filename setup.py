"""Package configuration."""
from setuptools import find_packages, setup

setup(
    name="timeseries",
    extras_require=dict(tests=['pytest']),
    version="0.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
