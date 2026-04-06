"""Tests for clustering evaluation metrics."""

import numpy as np
import pytest
from qasef.metrics import (
    clustering_accuracy, nmi, purity, ari, fscore, precision, evaluate_all,
)


class TestClusteringAccuracy:
    def test_perfect_match(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        assert clustering_accuracy(y, y) == pytest.approx(1.0)

    def test_permuted_labels(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([2, 2, 0, 0, 1, 1])
        assert clustering_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_worst_case(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        assert clustering_accuracy(y_true, y_pred) == pytest.approx(2 / 3)

    def test_single_cluster(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        assert clustering_accuracy(y_true, y_pred) == pytest.approx(1.0)


class TestNMI:
    def test_perfect(self):
        y = np.array([0, 0, 1, 1])
        assert nmi(y, y) == pytest.approx(1.0)

    def test_random_is_low(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 5, size=1000)
        y_pred = rng.integers(0, 5, size=1000)
        assert nmi(y_true, y_pred) < 0.1

    def test_permutation_invariant(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([2, 2, 0, 0, 1, 1])
        assert nmi(y_true, y_pred) == pytest.approx(1.0)


class TestPurity:
    def test_perfect(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        assert purity(y, y) == pytest.approx(1.0)

    def test_mixed(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        # Single cluster: majority class is 0 or 1 (both have 2), purity = 2/4
        assert purity(y_true, y_pred) == pytest.approx(0.5)

    def test_always_at_least_majority(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 3, size=100)
        y_pred = np.zeros(100, dtype=int)
        majority_frac = np.bincount(y_true).max() / 100
        assert purity(y_true, y_pred) == pytest.approx(majority_frac)


class TestARI:
    def test_perfect(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        assert ari(y, y) == pytest.approx(1.0)

    def test_random_near_zero(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 5, size=1000)
        y_pred = rng.integers(0, 5, size=1000)
        assert abs(ari(y_true, y_pred)) < 0.05


class TestFscore:
    def test_perfect(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        assert fscore(y, y) == pytest.approx(1.0)

    def test_permuted(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 1, 2, 2, 0, 0])
        assert fscore(y_true, y_pred) == pytest.approx(1.0)

    def test_range(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 3, size=100)
        y_pred = rng.integers(0, 3, size=100)
        score = fscore(y_true, y_pred)
        assert 0.0 <= score <= 1.0


class TestPrecision:
    def test_perfect(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        assert precision(y, y) == pytest.approx(1.0)

    def test_range(self):
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 4, size=100)
        y_pred = rng.integers(0, 4, size=100)
        score = precision(y_true, y_pred)
        assert 0.0 <= score <= 1.0


class TestEvaluateAll:
    def test_returns_all_keys(self):
        y = np.array([0, 0, 1, 1])
        result = evaluate_all(y, y)
        assert set(result.keys()) == {"ACC", "NMI", "Purity", "ARI", "F-score", "Precision"}

    def test_perfect_all_ones(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        result = evaluate_all(y, y)
        for metric, val in result.items():
            assert val == pytest.approx(1.0), f"{metric} should be 1.0"
