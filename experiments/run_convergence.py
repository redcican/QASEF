"""Experiment: Convergence analysis (Section 5.5 in the paper).

Tracks objective value and clustering metrics across iterations.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qasef import QASEF
from qasef.metrics import evaluate_all
from qasef.utils import generate_missing_mask, add_gaussian_noise
from data.loader import load_dataset

DATASETS = ["Handwritten", "BDGP", "Scene-15"]

N_CLUSTERS = {
    "Handwritten": 10, "BDGP": 5, "Scene-15": 15,
}


def run_convergence(dataset_name: str, data_dir: str | None = None,
                    missing_rate: float = 0.5, noise_sigma: float = 0.5):
    views, labels = load_dataset(dataset_name, data_dir)
    c = N_CLUSTERS[dataset_name]
    n = views[0].shape[0]
    V = len(views)
    m = min(c + 50, n // 2)

    rng = np.random.default_rng(0)

    # Apply degradation
    if noise_sigma > 0:
        views_input = add_gaussian_noise(views, noise_sigma, rng)
    else:
        views_input = views

    if missing_rate > 0:
        M = generate_missing_mask(n, V, missing_rate, rng)
    else:
        M = None

    # Run with verbose to track convergence
    model = QASEF(
        n_clusters=c, n_anchors=m, k_anchors=5,
        lam=1.0, max_iter=100, tol=1e-7,  # tighter tolerance to see full curve
        random_state=0, verbose=True,
    )
    pred = model.fit_predict(views_input, M)

    metrics = evaluate_all(labels, pred)
    print(f"\n  Final: ACC={metrics['ACC']:.4f}  NMI={metrics['NMI']:.4f}  "
          f"ARI={metrics['ARI']:.4f}")
    print(f"  Converged in {len(model.objectives_)} iterations")
    print(f"  Objective values: {model.objectives_[:5]} ... {model.objectives_[-3:]}")

    return model.objectives_, metrics


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    datasets = sys.argv[2:] if len(sys.argv) > 2 else DATASETS

    for ds in datasets:
        if ds not in N_CLUSTERS:
            print(f"Unknown dataset: {ds}, skipping.")
            continue

        for setting_name, rho, sigma in [
            ("Complete", 0.0, 0.0),
            ("Joint (50%, 0.5)", 0.5, 0.5),
        ]:
            print(f"\n{'='*50}")
            print(f"Dataset: {ds}, Setting: {setting_name}")
            print(f"{'='*50}")
            try:
                run_convergence(ds, data_dir, rho, sigma)
            except FileNotFoundError as e:
                print(f"  Skipped: {e}")
