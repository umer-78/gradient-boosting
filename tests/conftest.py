from pathlib import Path

import numpy as np
import pytest

from gboost import read_csv

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def housing():
    return read_csv(DATA / "housing.csv")


@pytest.fixture(scope="session")
def churn():
    return read_csv(DATA / "churn.csv")


@pytest.fixture(scope="session")
def linear():
    """y = 3a - 2b + noise, plus two columns that carry nothing."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(900, 4))
    y = 3 * x[:, 0] - 2 * x[:, 1] + rng.normal(0, 0.4, 900)
    return x, y
