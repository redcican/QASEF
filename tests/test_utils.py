"""Tests for data degradation utilities."""

import numpy as np
import pytest
from qasef.utils import generate_missing_mask, add_gaussian_noise


class TestGenerateMissingMask:
    def test_shape(self):
        M = generate_missing_mask(100, 4, 0.3, np.random.default_rng(0))
        assert M.shape == (100, 4)

    def test_zero_rate_all_ones(self):
        M = generate_missing_mask(50, 3, 0.0)
        assert np.all(M == 1)

    def test_values_binary(self):
        M = generate_missing_mask(100, 4, 0.5, np.random.default_rng(0))
        assert set(np.unique(M)).issubset({0.0, 1.0})

    def test_at_least_one_view_per_sample(self):
        for rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
            M = generate_missing_mask(200, 5, rate, np.random.default_rng(42))
            assert np.all(M.sum(axis=1) >= 1), \
                f"Sample with no views at rate={rate}"

    def test_approximate_missing_rate(self):
        n, V = 1000, 4
        rate = 0.3
        M = generate_missing_mask(n, V, rate, np.random.default_rng(0))
        actual_rate = 1 - M.mean()
        # Allow some slack due to the at-least-one constraint
        assert abs(actual_rate - rate) < 0.05

    def test_high_rate_with_two_views(self):
        # With V=2, max missing per sample is 1, so max rate is 50%
        M = generate_missing_mask(100, 2, 0.8, np.random.default_rng(0))
        assert np.all(M.sum(axis=1) >= 1)

    def test_reproducible(self):
        M1 = generate_missing_mask(50, 3, 0.4, np.random.default_rng(99))
        M2 = generate_missing_mask(50, 3, 0.4, np.random.default_rng(99))
        np.testing.assert_array_equal(M1, M2)


class TestAddGaussianNoise:
    def test_output_shape_preserved(self):
        views = [np.ones((10, 5)), np.ones((10, 3))]
        noisy = add_gaussian_noise(views, 0.1, np.random.default_rng(0))
        assert len(noisy) == 2
        assert noisy[0].shape == (10, 5)
        assert noisy[1].shape == (10, 3)

    def test_zero_noise_unchanged(self):
        views = [np.ones((10, 5))]
        noisy = add_gaussian_noise(views, 0.0, np.random.default_rng(0))
        np.testing.assert_array_equal(noisy[0], views[0])

    def test_noise_scales_with_sigma(self):
        rng = np.random.default_rng(0)
        views = [rng.standard_normal((100, 10))]
        noisy_low = add_gaussian_noise(views, 0.1, np.random.default_rng(1))
        noisy_high = add_gaussian_noise(views, 1.0, np.random.default_rng(1))
        diff_low = np.std(noisy_low[0] - views[0])
        diff_high = np.std(noisy_high[0] - views[0])
        assert diff_high > diff_low * 3  # 10x sigma → much more noise

    def test_does_not_modify_input(self):
        views = [np.ones((10, 5))]
        original = views[0].copy()
        add_gaussian_noise(views, 0.5, np.random.default_rng(0))
        np.testing.assert_array_equal(views[0], original)

    def test_reproducible(self):
        views = [np.ones((10, 5))]
        n1 = add_gaussian_noise(views, 0.5, np.random.default_rng(7))
        n2 = add_gaussian_noise(views, 0.5, np.random.default_rng(7))
        np.testing.assert_array_equal(n1[0], n2[0])
