"""Anchor graph construction and spectral embedding extraction.

Implements:
- k-means anchor selection
- Bipartite anchor graph (Eq. 6-7 in the paper)
- Spectral embedding via truncated SVD (Proposition 3.1)
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds


def select_anchors(X: np.ndarray, m: int, random_state: int = 0) -> np.ndarray:
    """Select m anchors via k-means.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Data matrix (only available samples).
    m : int
        Number of anchors.
    random_state : int
        Random seed for k-means.

    Returns
    -------
    anchors : np.ndarray of shape (m, d)
        Anchor points (cluster centers).
    """
    km = KMeans(n_clusters=m, n_init=1, max_iter=50, random_state=random_state)
    km.fit(X)
    return km.cluster_centers_


def build_anchor_graph(X: np.ndarray, anchors: np.ndarray, k: int = 5) -> np.ndarray:
    """Construct the bipartite anchor graph B via Eq. (7).

    For each sample x_i, connect to its k nearest anchors with weights
    computed in closed form.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Data matrix.
    anchors : np.ndarray of shape (m, d)
        Anchor points.
    k : int
        Number of nearest anchors per sample.

    Returns
    -------
    B : np.ndarray of shape (n, m)
        Bipartite graph matrix with non-negative entries summing to 1 per row.
    """
    n = X.shape[0]
    m = anchors.shape[0]
    k = min(k, m)

    # Compute squared distances: (n, m)
    # ||x_i - a_j||^2 = ||x_i||^2 - 2 x_i^T a_j + ||a_j||^2
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    A_sq = np.sum(anchors ** 2, axis=1, keepdims=True).T  # (1, m)
    dist_sq = X_sq - 2 * X @ anchors.T + A_sq  # (n, m)

    B = np.zeros((n, m), dtype=np.float64)

    # For each sample, find k nearest anchors
    if k < m:
        knn_indices = np.argpartition(dist_sq, k, axis=1)[:, :k]  # (n, k)
    else:
        knn_indices = np.tile(np.arange(m), (n, 1))  # all anchors

    for i in range(n):
        idx = knn_indices[i]
        dists = dist_sq[i, idx]

        # Sort to identify the k-th nearest
        order = np.argsort(dists)
        idx = idx[order]
        dists = dists[order]

        # Distance to the (k+1)-th nearest anchor for normalization
        if k < m:
            all_sorted = np.argpartition(dist_sq[i], k)
            d_kp1 = dist_sq[i, all_sorted[k]]
        else:
            d_kp1 = dists[-1] * 2 + 1  # fallback

        # Closed-form solution (Eq. 7)
        denom = k * d_kp1 - dists.sum()
        if denom < 1e-12:
            # Equidistant anchors: uniform weights
            B[i, idx] = 1.0 / k
        else:
            weights = (d_kp1 - dists) / denom
            weights = np.maximum(weights, 0)
            B[i, idx] = weights

    return B


def extract_embedding(B: np.ndarray, c: int) -> np.ndarray:
    """Extract spectral embedding from anchor graph via truncated SVD.

    Computes F^(v) = top-c left singular vectors of S_tilde = B D^{-1/2}.

    Parameters
    ----------
    B : np.ndarray of shape (n, m)
        Bipartite anchor graph.
    c : int
        Number of clusters (embedding dimension).

    Returns
    -------
    F : np.ndarray of shape (n, c)
        Spectral embedding matrix.
    """
    # Degree matrix D: d_j = sum_i B[i, j]
    d = B.sum(axis=0)  # (m,)

    # Avoid division by zero
    d_inv_sqrt = np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0)

    # S_tilde = B @ diag(d^{-1/2}) of shape (n, m)
    S_tilde = B * d_inv_sqrt[np.newaxis, :]

    # Truncated SVD: top c singular triplets
    # svds returns in ascending order of singular values
    c_actual = min(c, min(S_tilde.shape) - 1)
    if c_actual < 1:
        c_actual = 1

    U, sigma, Vt = svds(S_tilde, k=c_actual)

    # svds returns in ascending order; reverse to descending
    order = np.argsort(-sigma)
    F = U[:, order]

    # Pad if needed (when m < c)
    if F.shape[1] < c:
        pad = np.zeros((F.shape[0], c - F.shape[1]))
        F = np.concatenate([F, pad], axis=1)

    return F


def compute_all_embeddings(views: list[np.ndarray], M: np.ndarray,
                           c: int, m: int, k: int = 5,
                           random_state: int = 0) -> list[np.ndarray]:
    """Compute spectral embeddings for all views.

    Parameters
    ----------
    views : list of np.ndarray
        List of V data matrices, each of shape (n, d_v).
    M : np.ndarray of shape (n, V)
        Availability mask.
    c : int
        Number of clusters.
    m : int
        Number of anchors per view.
    k : int
        Number of nearest anchors.
    random_state : int
        Random seed.

    Returns
    -------
    embeddings : list of np.ndarray
        List of V embedding matrices, each of shape (n, c).
    """
    V = len(views)
    n = views[0].shape[0]
    embeddings = []

    for v in range(V):
        available = M[:, v] > 0.5
        X_avail = views[v][available]

        if X_avail.shape[0] < m:
            m_v = max(c + 1, X_avail.shape[0] // 2)
        else:
            m_v = m

        anchors = select_anchors(X_avail, m_v, random_state=random_state)
        B_avail = build_anchor_graph(X_avail, anchors, k=k)

        # Build full B with zeros for missing samples
        B_full = np.zeros((n, m_v), dtype=np.float64)
        B_full[available] = B_avail

        F = extract_embedding(B_full, c)

        # Zero out rows for missing samples (Eq. 12)
        F[~available] = 0.0

        embeddings.append(F)

    return embeddings
