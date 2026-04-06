"""Clustering evaluation metrics: ACC, NMI, Purity, ARI, F-score, Precision."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering accuracy via the Hungarian algorithm.

    Finds the optimal one-to-one mapping between predicted cluster labels
    and ground-truth labels that maximizes accuracy.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    n = y_true.shape[0]

    labels_true = set(y_true)
    labels_pred = set(y_pred)
    n_classes = max(len(labels_true), len(labels_pred))

    # Build cost matrix
    cost = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n):
        cost[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)
    return cost[row_ind, col_ind].sum() / n


def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mutual information (arithmetic average)."""
    return normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cluster purity."""
    n = y_true.shape[0]
    total = 0
    for cluster_id in np.unique(y_pred):
        mask = y_pred == cluster_id
        counts = np.bincount(y_true[mask])
        total += counts.max()
    return total / n


def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Adjusted Rand index."""
    return adjusted_rand_score(y_true, y_pred)


def _contingency(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Contingency matrix."""
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    C = np.zeros((clusters.shape[0], classes.shape[0]), dtype=np.int64)
    cluster_map = {c: i for i, c in enumerate(clusters)}
    class_map = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        C[cluster_map[p], class_map[t]] += 1
    return C


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering precision via Hungarian matching."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    n = y_true.shape[0]

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    n_classes = max(labels_true.max() + 1, labels_pred.max() + 1)

    cost = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n):
        cost[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)

    # Build mapping
    mapping = dict(zip(row_ind, col_ind))
    mapped_pred = np.array([mapping.get(p, -1) for p in y_pred])

    # Precision: for each predicted cluster, fraction of correct assignments
    prec_sum = 0.0
    count = 0
    for cluster_id in np.unique(y_pred):
        mask = y_pred == cluster_id
        n_k = mask.sum()
        if n_k == 0:
            continue
        matched_label = mapping.get(cluster_id, -1)
        tp = np.sum(y_true[mask] == matched_label)
        prec_sum += tp / n_k
        count += 1
    return prec_sum / count if count > 0 else 0.0


def fscore(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F-score (macro-averaged) via Hungarian matching."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    n = y_true.shape[0]

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    n_classes = max(labels_true.max() + 1, labels_pred.max() + 1)

    cost = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n):
        cost[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = dict(zip(row_ind, col_ind))
    mapped_pred = np.array([mapping.get(p, -1) for p in y_pred])

    f_sum = 0.0
    count = 0
    for label in np.unique(y_true):
        tp = np.sum((mapped_pred == label) & (y_true == label))
        fp = np.sum((mapped_pred == label) & (y_true != label))
        fn = np.sum((mapped_pred != label) & (y_true == label))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f_sum += f
        count += 1
    return f_sum / count if count > 0 else 0.0


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all six metrics and return as a dictionary."""
    return {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": nmi(y_true, y_pred),
        "Purity": purity(y_true, y_pred),
        "ARI": ari(y_true, y_pred),
        "F-score": fscore(y_true, y_pred),
        "Precision": precision(y_true, y_pred),
    }
