"""Experiment: Ablation study (Section 5.4 in the paper).

Compares five variants:
  - QASEF (full): complete model
  - w/o Q: quality scores fixed to M (no quality estimation)
  - w/o R: lambda=0 (no regularization)
  - Fixed-Q: quality scores frozen after initialization
  - Sim-Q: quality from inter-view embedding similarity
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qasef import QASEF
from qasef.metrics import evaluate_all
from qasef.utils import generate_missing_mask, add_gaussian_noise
from data.loader import load_dataset

DATASETS = [
    "ORL", "100leaves", "Handwritten", "Caltech101-20", "BDGP",
    "Scene-15", "Caltech101",
]

N_CLUSTERS = {
    "ORL": 40, "100leaves": 100, "Handwritten": 10, "Caltech101-20": 20,
    "BDGP": 5, "Scene-15": 15, "Caltech101": 101,
}

VARIANTS = {
    "QASEF": {"quality_mode": "full"},
    "w/o Q": {"quality_mode": "none"},
    "w/o R": {"quality_mode": "no_reg"},
    "Fixed-Q": {"quality_mode": "fixed"},
    "Sim-Q": {"quality_mode": "similarity"},
}

N_RUNS = 10
MISSING_RATE = 0.5
NOISE_SIGMA = 0.5


def run_ablation(dataset_name: str, data_dir: str | None = None):
    views, labels = load_dataset(dataset_name, data_dir)
    c = N_CLUSTERS[dataset_name]
    n = views[0].shape[0]
    V = len(views)
    m = min(c + 50, n // 2)

    print(f"\n  Joint degradation: missing={MISSING_RATE*100:.0f}%, "
          f"noise sigma={NOISE_SIGMA}")

    for variant_name, kwargs in VARIANTS.items():
        print(f"\n  Variant: {variant_name}")
        all_results = []
        for run in range(N_RUNS):
            rng = np.random.default_rng(run)
            noisy_views = add_gaussian_noise(views, NOISE_SIGMA, rng)
            M = generate_missing_mask(n, V, MISSING_RATE, rng)

            model = QASEF(
                n_clusters=c, n_anchors=m, k_anchors=5,
                lam=1.0, max_iter=100, tol=1e-5,
                random_state=run, verbose=False,
                **kwargs,
            )
            pred = model.fit_predict(noisy_views, M)
            metrics = evaluate_all(labels, pred)
            all_results.append(metrics)

        print(f"    {'Metric':<12} {'Mean':>8} {'Std':>8}")
        print(f"    {'-'*30}")
        for metric in ["ACC", "NMI", "Purity", "ARI", "F-score", "Precision"]:
            vals = [r[metric] for r in all_results]
            print(f"    {metric:<12} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    datasets = sys.argv[2:] if len(sys.argv) > 2 else DATASETS

    for ds in datasets:
        if ds not in N_CLUSTERS:
            print(f"Unknown dataset: {ds}, skipping.")
            continue
        print(f"\n{'='*50}")
        print(f"Dataset: {ds}")
        print(f"{'='*50}")
        try:
            run_ablation(ds, data_dir)
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
