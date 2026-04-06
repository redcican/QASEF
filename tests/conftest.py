"""Shared fixtures for QASEF tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_data(rng):
    """Well-separated 3-view dataset: 200 samples, 4 clusters."""
    n, c, V = 200, 4, 3
    labels = np.repeat(np.arange(c), n // c)
    views = []
    for _ in range(V):
        d = rng.integers(10, 20)
        centers = rng.standard_normal((c, d)) * 5
        X = np.vstack([centers[l] + rng.standard_normal(d) * 0.3
                       for l in labels])
        views.append(X)
    return views, labels, c


@pytest.fixture
def small_data(rng):
    """Tiny dataset for fast unit tests: 40 samples, 2 clusters, 2 views."""
    n, c, V = 40, 2, 2
    labels = np.repeat(np.arange(c), n // c)
    views = []
    for _ in range(V):
        d = 5
        centers = rng.standard_normal((c, d)) * 4
        X = np.vstack([centers[l] + rng.standard_normal(d) * 0.5
                       for l in labels])
        views.append(X)
    return views, labels, c
