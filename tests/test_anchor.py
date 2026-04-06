"""Tests for anchor graph construction and spectral embedding extraction."""

import numpy as np
import pytest
from qasef.anchor import (
    select_anchors, build_anchor_graph, extract_embedding,
    compute_all_embeddings,
)


class TestSelectAnchors:
    def test_shape(self):
        X = np.random.default_rng(0).standard_normal((100, 10))
        anchors = select_anchors(X, m=15)
        assert anchors.shape == (15, 10)

    def test_anchors_within_data_range(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 5))
        anchors = select_anchors(X, m=20)
        for d in range(5):
            assert anchors[:, d].min() >= X[:, d].min() - 1
            assert anchors[:, d].max() <= X[:, d].max() + 1

    def test_reproducible(self):
        X = np.random.default_rng(0).standard_normal((100, 10))
        a1 = select_anchors(X, 10, random_state=42)
        a2 = select_anchors(X, 10, random_state=42)
        np.testing.assert_array_equal(a1, a2)


class TestBuildAnchorGraph:
    def test_shape(self):
        X = np.random.default_rng(0).standard_normal((50, 5))
        anchors = select_anchors(X, 10)
        B = build_anchor_graph(X, anchors, k=3)
        assert B.shape == (50, 10)

    def test_rows_sum_to_one(self):
        X = np.random.default_rng(0).standard_normal((100, 8))
        anchors = select_anchors(X, 15)
        B = build_anchor_graph(X, anchors, k=5)
        np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-10)

    def test_non_negative(self):
        X = np.random.default_rng(0).standard_normal((100, 8))
        anchors = select_anchors(X, 15)
        B = build_anchor_graph(X, anchors, k=5)
        assert np.all(B >= -1e-12)

    def test_sparse_k_connections(self):
        X = np.random.default_rng(0).standard_normal((100, 8))
        anchors = select_anchors(X, 20)
        k = 3
        B = build_anchor_graph(X, anchors, k=k)
        # Each row should have at most k non-zero entries
        for i in range(100):
            assert np.count_nonzero(B[i]) <= k

    def test_k_equals_m(self):
        X = np.random.default_rng(0).standard_normal((50, 5))
        m = 8
        anchors = select_anchors(X, m)
        B = build_anchor_graph(X, anchors, k=m)
        assert B.shape == (50, m)
        np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-10)


class TestExtractEmbedding:
    def test_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 10))
        anchors = select_anchors(X, 20)
        B = build_anchor_graph(X, anchors, k=5)
        F = extract_embedding(B, c=4)
        assert F.shape == (100, 4)

    def test_orthonormal_columns(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 10))
        anchors = select_anchors(X, 30)
        B = build_anchor_graph(X, anchors, k=5)
        F = extract_embedding(B, c=5)
        # F^T F should be close to I
        np.testing.assert_allclose(F.T @ F, np.eye(5), atol=1e-8)

    def test_embedding_captures_structure(self):
        # Two well-separated clusters → first embedding dimension should separate them
        rng = np.random.default_rng(0)
        n = 100
        X = np.vstack([
            rng.standard_normal((n // 2, 5)) + 10,
            rng.standard_normal((n // 2, 5)) - 10,
        ])
        anchors = select_anchors(X, 10)
        B = build_anchor_graph(X, anchors, k=3)
        F = extract_embedding(B, c=2)
        # The two groups should have different signs in the 2nd column
        # (1st singular vector is roughly constant)
        group1 = F[:n // 2, 1]
        group2 = F[n // 2:, 1]
        assert abs(group1.mean() - group2.mean()) > 0.05


class TestComputeAllEmbeddings:
    def test_output_structure(self):
        rng = np.random.default_rng(0)
        n, V, c = 60, 3, 3
        views = [rng.standard_normal((n, d)) for d in [10, 8, 12]]
        M = np.ones((n, V))
        embeddings = compute_all_embeddings(views, M, c=c, m=15, k=3)
        assert len(embeddings) == V
        for F in embeddings:
            assert F.shape == (n, c)

    def test_missing_rows_are_zero(self):
        rng = np.random.default_rng(0)
        n, V, c = 60, 2, 3
        views = [rng.standard_normal((n, 10)) for _ in range(V)]
        M = np.ones((n, V))
        M[5, 0] = 0
        M[10, 1] = 0
        M[15, 0] = 0
        embeddings = compute_all_embeddings(views, M, c=c, m=15, k=3)
        np.testing.assert_array_equal(embeddings[0][5], 0.0)
        np.testing.assert_array_equal(embeddings[1][10], 0.0)
        np.testing.assert_array_equal(embeddings[0][15], 0.0)

    def test_all_views_available_no_zeros(self):
        rng = np.random.default_rng(0)
        n, V, c = 60, 2, 3
        views = [rng.standard_normal((n, 10)) for _ in range(V)]
        M = np.ones((n, V))
        embeddings = compute_all_embeddings(views, M, c=c, m=15, k=3)
        for F in embeddings:
            # At least most rows should be non-zero
            zero_rows = np.all(F == 0, axis=1).sum()
            assert zero_rows == 0
