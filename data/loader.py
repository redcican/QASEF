"""Dataset loader for multi-view clustering benchmarks.

Supports .mat files with standard multi-view dataset formats.
"""

import os
import numpy as np
from scipy.io import loadmat


def load_dataset(name: str, data_dir: str | None = None) -> tuple:
    """Load a multi-view benchmark dataset from a .mat file.

    Parameters
    ----------
    name : str
        Dataset name (e.g., 'Handwritten', 'BDGP', 'ORL').
    data_dir : str, optional
        Directory containing .mat files. Defaults to this file's directory.

    Returns
    -------
    views : list of np.ndarray
        List of V data matrices, each of shape (n, d_v).
    labels : np.ndarray of shape (n,)
        Ground-truth cluster labels (0-indexed).
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))

    filepath = os.path.join(data_dir, f"{name}.mat")
    if not os.path.exists(filepath):
        # Try lowercase
        filepath = os.path.join(data_dir, f"{name.lower()}.mat")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file not found: {name}.mat in {data_dir}"
        )

    data = loadmat(filepath)

    views = []
    labels = None

    # Try common .mat key patterns for multi-view data
    # Pattern 1: X cell array + Y labels
    if "X" in data:
        X = data["X"]
        if hasattr(X, "shape") and X.ndim == 2 and X.shape[0] == 1:
            # Cell array stored as (1, V) object array
            for v in range(X.shape[1]):
                view = np.array(X[0, v], dtype=np.float64)
                if view.ndim == 1:
                    view = view.reshape(-1, 1)
                views.append(view)
        elif hasattr(X, "shape") and X.shape[1] == 1:
            for v in range(X.shape[0]):
                view = np.array(X[v, 0], dtype=np.float64)
                if view.ndim == 1:
                    view = view.reshape(-1, 1)
                views.append(view)

    # Pattern 2: x1, x2, x3, ...  or  X1, X2, X3, ...
    if not views:
        v = 1
        while True:
            for prefix in ["x", "X", "view", "View"]:
                key = f"{prefix}{v}"
                if key in data:
                    view = np.array(data[key], dtype=np.float64)
                    if view.ndim == 1:
                        view = view.reshape(-1, 1)
                    views.append(view)
                    break
            else:
                break
            v += 1

    # Pattern 3: data cell array
    if not views and "data" in data:
        d = data["data"]
        if hasattr(d, "shape"):
            dim = max(d.shape)
            for v in range(dim):
                idx = (0, v) if d.shape[0] == 1 else (v, 0)
                view = np.array(d[idx], dtype=np.float64)
                if view.ndim == 1:
                    view = view.reshape(-1, 1)
                views.append(view)

    if not views:
        raise ValueError(
            f"Could not parse views from {name}.mat. "
            f"Available keys: {[k for k in data.keys() if not k.startswith('__')]}"
        )

    # Load labels
    for key in ["Y", "y", "gt", "gnd", "labels", "label", "truelabel"]:
        if key in data:
            labels = np.array(data[key], dtype=np.int64).ravel()
            break

    if labels is None:
        raise ValueError(
            f"Could not find labels in {name}.mat. "
            f"Available keys: {[k for k in data.keys() if not k.startswith('__')]}"
        )

    # Ensure 0-indexed labels
    if labels.min() >= 1:
        labels = labels - labels.min()

    # Ensure all views have the same number of samples
    n = views[0].shape[0]
    for v, view in enumerate(views):
        if view.shape[0] != n:
            raise ValueError(
                f"View {v} has {view.shape[0]} samples, expected {n}"
            )

    return views, labels


def list_datasets(data_dir: str | None = None) -> list[str]:
    """List available .mat dataset files."""
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))

    datasets = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".mat"):
            datasets.append(f[:-4])
    return datasets
