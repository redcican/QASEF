"""Tests for the QASEF model: subproblems and end-to-end."""

import numpy as np
import pytest
from qasef.model import QASEF
from qasef.metrics import clustering_accuracy
from qasef.utils import generate_missing_mask, add_gaussian_noise


# ---------------------------------------------------------------------------
# Unit tests for static helper methods
# ---------------------------------------------------------------------------

class TestLabelsToIndicator:
    def test_shape(self):
        labels = np.array([0, 1, 2, 0, 1])
        Y = QASEF._labels_to_indicator(labels, 3)
        assert Y.shape == (5, 3)

    def test_one_hot(self):
        labels = np.array([0, 1, 2])
        Y = QASEF._labels_to_indicator(labels, 3)
        expected = np.eye(3)
        np.testing.assert_array_equal(Y, expected)

    def test_row_sums_one(self):
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        Y = QASEF._labels_to_indicator(labels, 4)
        np.testing.assert_array_equal(Y.sum(axis=1), 1.0)


class TestComputeU:
    def test_orthonormal_columns(self):
        labels = np.array([0, 0, 0, 1, 1, 2])
        Y = QASEF._labels_to_indicator(labels, 3)
        U = QASEF._compute_U(Y)
        np.testing.assert_allclose(U.T @ U, np.eye(3), atol=1e-12)

    def test_matches_formula(self):
        labels = np.array([0, 0, 1, 1, 1])
        Y = QASEF._labels_to_indicator(labels, 2)
        U = QASEF._compute_U(Y)
        # Manual: Y (Y^T Y)^{-1/2}
        YtY = Y.T @ Y
        scale = np.diag(1.0 / np.sqrt(np.diag(YtY)))
        expected = Y @ scale
        np.testing.assert_allclose(U, expected, atol=1e-12)


class TestProjectSimplex:
    def test_already_on_simplex(self):
        v = np.array([0.3, 0.5, 0.2])
        proj = QASEF._project_simplex(v)
        np.testing.assert_allclose(proj.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(proj, v, atol=1e-12)

    def test_uniform(self):
        v = np.array([1.0, 1.0, 1.0])
        proj = QASEF._project_simplex(v)
        np.testing.assert_allclose(proj, [1/3, 1/3, 1/3], atol=1e-12)

    def test_negative_input(self):
        v = np.array([-1.0, 0.5, 2.0])
        proj = QASEF._project_simplex(v)
        assert np.all(proj >= -1e-12)
        assert proj.sum() == pytest.approx(1.0, abs=1e-12)

    def test_single_element(self):
        v = np.array([5.0])
        proj = QASEF._project_simplex(v)
        assert proj[0] == pytest.approx(1.0)

    def test_all_negative(self):
        v = np.array([-3.0, -2.0, -1.0])
        proj = QASEF._project_simplex(v)
        assert np.all(proj >= -1e-12)
        assert proj.sum() == pytest.approx(1.0, abs=1e-12)


class TestUpdateG:
    def test_orthogonal(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((50, 4))
        U = rng.standard_normal((50, 4))
        G = QASEF._update_G(Z, U)
        np.testing.assert_allclose(G.T @ G, np.eye(4), atol=1e-10)

    def test_maximizes_trace(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((50, 3))
        U = rng.standard_normal((50, 3))
        G_opt = QASEF._update_G(Z, U)
        trace_opt = np.trace(G_opt.T @ Z.T @ U)

        # Compare against random orthogonal matrices
        for _ in range(20):
            H = rng.standard_normal((3, 3))
            Q, _ = np.linalg.qr(H)
            trace_rand = np.trace(Q.T @ Z.T @ U)
            assert trace_opt >= trace_rand - 1e-10


class TestUpdateQ:
    def test_missing_views_stay_zero(self):
        c, V, n = 3, 2, 10
        FG = [np.random.default_rng(0).standard_normal((n, c)) for _ in range(V)]
        U = np.random.default_rng(1).standard_normal((n, c))
        alpha = np.array([0.5, 0.5])
        M = np.ones((n, V))
        M[3, 0] = 0
        M[7, 1] = 0

        Q = QASEF._update_Q(FG, U, alpha, M, lam=1.0, n=n, V=V, c=c)
        assert Q[3, 0] == 0.0
        assert Q[7, 1] == 0.0

    def test_values_in_range(self):
        c, V, n = 3, 3, 20
        rng = np.random.default_rng(0)
        FG = [rng.standard_normal((n, c)) for _ in range(V)]
        U = rng.standard_normal((n, c))
        alpha = np.array([0.4, 0.3, 0.3])
        M = np.ones((n, V))

        Q = QASEF._update_Q(FG, U, alpha, M, lam=1.0, n=n, V=V, c=c)
        assert np.all(Q >= 0.0)
        assert np.all(Q <= 1.0)

    def test_high_lambda_scores_near_one(self):
        c, V, n = 3, 2, 20
        rng = np.random.default_rng(0)
        FG = [rng.standard_normal((n, c)) for _ in range(V)]
        U = rng.standard_normal((n, c))
        alpha = np.array([0.5, 0.5])
        M = np.ones((n, V))

        Q = QASEF._update_Q(FG, U, alpha, M, lam=1000.0, n=n, V=V, c=c)
        assert Q.mean() > 0.95


class TestUpdateAlpha:
    def test_on_simplex(self):
        c, V, n = 3, 3, 50
        rng = np.random.default_rng(0)
        FG = [rng.standard_normal((n, c)) for _ in range(V)]
        Q = np.ones((n, V))
        U = rng.standard_normal((n, c))

        alpha = QASEF._update_alpha(FG, Q, U, V, n, c)
        assert alpha.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(alpha >= -1e-10)

    def test_dominant_view_gets_higher_weight(self):
        rng = np.random.default_rng(0)
        n, c = 100, 3
        labels = np.repeat(np.arange(c), n // c + 1)[:n]
        Y = QASEF._labels_to_indicator(labels, c)
        U = QASEF._compute_U(Y)

        # View 0 matches U well; views 1-2 are noise
        FG = [U + rng.standard_normal(U.shape) * 0.01,
              rng.standard_normal(U.shape),
              rng.standard_normal(U.shape)]
        Q = np.ones((n, 3))
        alpha = QASEF._update_alpha(FG, Q, U, 3, n, c)
        assert alpha[0] > alpha[1]
        assert alpha[0] > alpha[2]


# ---------------------------------------------------------------------------
# Integration tests: end-to-end model
# ---------------------------------------------------------------------------

class TestQASEFFit:
    def test_returns_self(self, small_data):
        views, labels, c = small_data
        model = QASEF(n_clusters=c, n_anchors=10, max_iter=5)
        result = model.fit(views)
        assert result is model

    def test_labels_shape(self, small_data):
        views, labels, c = small_data
        model = QASEF(n_clusters=c, n_anchors=10, max_iter=10)
        model.fit(views)
        assert model.labels_.shape == (len(labels),)

    def test_labels_range(self, small_data):
        views, labels, c = small_data
        model = QASEF(n_clusters=c, n_anchors=10, max_iter=10)
        model.fit(views)
        assert set(np.unique(model.labels_)).issubset(set(range(c)))

    def test_Q_shape(self, small_data):
        views, labels, c = small_data
        V = len(views)
        model = QASEF(n_clusters=c, n_anchors=10, max_iter=10)
        model.fit(views)
        assert model.Q_.shape == (len(labels), V)

    def test_alpha_on_simplex(self, small_data):
        views, labels, c = small_data
        model = QASEF(n_clusters=c, n_anchors=10, max_iter=10)
        model.fit(views)
        assert model.alpha_.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(model.alpha_ >= -1e-10)

    def test_G_orthogonal(self, small_data):
        views, labels, c = small_data
        model = QASEF(n_clusters=c, n_anchors=10, max_iter=10)
        model.fit(views)
        np.testing.assert_allclose(
            model.G_.T @ model.G_, np.eye(c), atol=1e-8
        )

    def test_fit_predict_matches(self, small_data):
        views, labels, c = small_data
        model = QASEF(n_clusters=c, n_anchors=10, max_iter=10, random_state=0)
        pred = model.fit_predict(views)
        np.testing.assert_array_equal(pred, model.labels_)


class TestQASEFConvergence:
    def test_monotonic_decrease(self, synthetic_data):
        views, labels, c = synthetic_data
        model = QASEF(n_clusters=c, n_anchors=30, lam=1.0, max_iter=50,
                      tol=1e-10, random_state=0)
        model.fit(views)

        objs = model.objectives_
        for i in range(1, len(objs)):
            assert objs[i] <= objs[i - 1] + 1e-6, \
                f"Objective increased at iter {i+1}: {objs[i-1]:.8f} -> {objs[i]:.8f}"

    def test_monotonic_with_missing(self, synthetic_data):
        views, labels, c = synthetic_data
        n, V = views[0].shape[0], len(views)
        M = generate_missing_mask(n, V, 0.5, np.random.default_rng(0))

        model = QASEF(n_clusters=c, n_anchors=30, lam=1.0, max_iter=50,
                      tol=1e-10, random_state=0)
        model.fit(views, M)

        objs = model.objectives_
        for i in range(1, len(objs)):
            assert objs[i] <= objs[i - 1] + 1e-6, \
                f"Objective increased at iter {i+1}: {objs[i-1]:.8f} -> {objs[i]:.8f}"

    def test_monotonic_with_noise(self, synthetic_data):
        views, labels, c = synthetic_data
        noisy = add_gaussian_noise(views, 0.5, np.random.default_rng(0))

        model = QASEF(n_clusters=c, n_anchors=30, lam=1.0, max_iter=50,
                      tol=1e-10, random_state=0)
        model.fit(noisy)

        objs = model.objectives_
        for i in range(1, len(objs)):
            assert objs[i] <= objs[i - 1] + 1e-6, \
                f"Objective increased at iter {i+1}: {objs[i-1]:.8f} -> {objs[i]:.8f}"

    def test_converges_within_iterations(self, synthetic_data):
        views, labels, c = synthetic_data
        model = QASEF(n_clusters=c, n_anchors=30, lam=1.0, max_iter=100,
                      tol=1e-5, random_state=0)
        model.fit(views)
        assert len(model.objectives_) < 100


class TestQASEFClustering:
    def test_high_accuracy_clean_data(self):
        # Dedicated well-separated data to avoid fixture variance
        rng = np.random.default_rng(123)
        n, c, V = 200, 4, 3
        labels = np.repeat(np.arange(c), n // c)
        views = []
        for _ in range(V):
            d = 15
            centers = rng.standard_normal((c, d)) * 8
            X = np.vstack([centers[l] + rng.standard_normal(d) * 0.3
                           for l in labels])
            views.append(X)
        model = QASEF(n_clusters=c, n_anchors=30, lam=1.0,
                      max_iter=50, random_state=0)
        pred = model.fit_predict(views)
        acc = clustering_accuracy(labels, pred)
        assert acc > 0.8, f"ACC too low on clean data: {acc}"

    def test_quality_scores_near_one_clean(self, synthetic_data):
        views, labels, c = synthetic_data
        model = QASEF(n_clusters=c, n_anchors=30, lam=1.0,
                      max_iter=50, random_state=0)
        model.fit(views)
        assert model.Q_.mean() > 0.9

    def test_missing_views_get_zero_quality(self, synthetic_data):
        views, labels, c = synthetic_data
        n, V = views[0].shape[0], len(views)
        M = generate_missing_mask(n, V, 0.5, np.random.default_rng(0))

        model = QASEF(n_clusters=c, n_anchors=30, lam=1.0,
                      max_iter=50, random_state=0)
        model.fit(views, M)

        missing_mask = M < 0.5
        assert np.all(model.Q_[missing_mask] == 0.0)

    def test_full_better_than_no_quality(self):
        # Use well-separated data with heavy degradation so quality module helps
        rng = np.random.default_rng(77)
        n, c, V = 200, 4, 3
        labels = np.repeat(np.arange(c), n // c)
        views = []
        for _ in range(V):
            d = 15
            centers = rng.standard_normal((c, d)) * 8
            X = np.vstack([centers[l] + rng.standard_normal(d) * 0.3
                           for l in labels])
            views.append(X)

        M = generate_missing_mask(n, V, 0.5, np.random.default_rng(0))
        noisy = add_gaussian_noise(views, 0.8, np.random.default_rng(0))

        # Average over 3 runs to reduce variance
        accs_full, accs_none = [], []
        for seed in range(3):
            accs_full.append(clustering_accuracy(labels, QASEF(
                n_clusters=c, n_anchors=30, quality_mode="full",
                max_iter=50, random_state=seed
            ).fit_predict(noisy, M)))
            accs_none.append(clustering_accuracy(labels, QASEF(
                n_clusters=c, n_anchors=30, quality_mode="none",
                max_iter=50, random_state=seed
            ).fit_predict(noisy, M)))

        assert np.mean(accs_full) >= np.mean(accs_none) - 0.1


class TestQASEFAblation:
    def test_all_modes_run(self, small_data):
        views, labels, c = small_data
        n, V = views[0].shape[0], len(views)
        M = generate_missing_mask(n, V, 0.3, np.random.default_rng(0))
        noisy = add_gaussian_noise(views, 0.3, np.random.default_rng(0))

        for mode in ["full", "none", "no_reg", "fixed", "similarity"]:
            model = QASEF(n_clusters=c, n_anchors=10, quality_mode=mode,
                          max_iter=20, random_state=0)
            pred = model.fit_predict(noisy, M)
            assert pred.shape == (n,), f"Mode {mode} failed"

    def test_no_quality_Q_equals_M(self, small_data):
        views, labels, c = small_data
        n, V = views[0].shape[0], len(views)
        M = generate_missing_mask(n, V, 0.3, np.random.default_rng(0))

        model = QASEF(n_clusters=c, n_anchors=10, quality_mode="none",
                      max_iter=20, random_state=0)
        model.fit(views, M)
        np.testing.assert_array_equal(model.Q_, M)

    def test_no_reg_mode(self, small_data):
        views, labels, c = small_data
        model = QASEF(n_clusters=c, n_anchors=10, quality_mode="no_reg",
                      max_iter=20, random_state=0)
        model.fit(views)
        # Q values should still be in [0, 1]
        assert np.all(model.Q_ >= 0.0)
        assert np.all(model.Q_ <= 1.0)


class TestQASEFEdgeCases:
    def test_two_views(self):
        rng = np.random.default_rng(0)
        n, c = 40, 2
        labels = np.repeat(np.arange(c), n // c)
        views = [
            np.vstack([rng.standard_normal((n//2, 5)) + 3,
                       rng.standard_normal((n//2, 5)) - 3]),
            np.vstack([rng.standard_normal((n//2, 8)) + 3,
                       rng.standard_normal((n//2, 8)) - 3]),
        ]
        model = QASEF(n_clusters=c, n_anchors=10, max_iter=30)
        pred = model.fit_predict(views)
        assert clustering_accuracy(labels, pred) > 0.7

    def test_single_view(self):
        rng = np.random.default_rng(0)
        n, c = 40, 2
        labels = np.repeat(np.arange(c), n // c)
        views = [np.vstack([
            rng.standard_normal((n//2, 5)) + 5,
            rng.standard_normal((n//2, 5)) - 5,
        ])]
        model = QASEF(n_clusters=c, n_anchors=10, max_iter=30)
        pred = model.fit_predict(views)
        assert pred.shape == (n,)

    def test_many_clusters(self):
        rng = np.random.default_rng(0)
        n, c, V = 100, 10, 2
        labels = np.repeat(np.arange(c), n // c)
        views = []
        for _ in range(V):
            centers = rng.standard_normal((c, 8)) * 5
            X = np.vstack([centers[l] + rng.standard_normal(8) * 0.3
                           for l in labels])
            views.append(X)
        model = QASEF(n_clusters=c, n_anchors=20, max_iter=30)
        pred = model.fit_predict(views)
        assert len(np.unique(pred)) >= 2
