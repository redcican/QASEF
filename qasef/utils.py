"""Utility functions for data degradation (missing views, noise)."""

import numpy as np


def generate_missing_mask(n: int, V: int, missing_rate: float,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a binary availability mask M of shape (n, V).

    Each sample is guaranteed to have at least one available view.

    Parameters
    ----------
    n : int
        Number of samples.
    V : int
        Number of views.
    missing_rate : float
        Fraction of sample-view pairs to mark as missing (0 to 1).
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    M : np.ndarray of shape (n, V)
        Binary mask where M[i, v] = 1 means view v is available for sample i.
    """
    if rng is None:
        rng = np.random.default_rng()

    M = np.ones((n, V), dtype=np.float64)

    if missing_rate <= 0:
        return M

    # Total entries to remove
    total_entries = n * V
    n_missing = int(total_entries * missing_rate)

    # Create all (i, v) indices, randomly mark some as missing
    all_indices = [(i, v) for i in range(n) for v in range(V)]
    rng.shuffle(all_indices)

    removed = 0
    for i, v in all_indices:
        if removed >= n_missing:
            break
        # Check that sample i would still have at least one view
        if M[i].sum() > 1:
            M[i, v] = 0
            removed += 1

    return M


def add_gaussian_noise(views: list[np.ndarray], noise_std: float,
                       rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Add Gaussian noise to multi-view data.

    Parameters
    ----------
    views : list of np.ndarray
        List of V data matrices, each of shape (n, d_v).
    noise_std : float
        Standard deviation of the Gaussian noise (relative to per-view std).
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    noisy_views : list of np.ndarray
        Noisy copies of the input views.
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy_views = []
    for X in views:
        view_std = X.std()
        noise = rng.normal(0, noise_std * view_std, size=X.shape)
        noisy_views.append(X + noise)
    return noisy_views
