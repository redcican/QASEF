"""Experiment: Parameter sensitivity analysis (Section 5.5 in the paper).

Studies the effect of:
  - Number of anchors m
  - Regularization parameter lambda
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

ANCHOR_VALUES = [20, 40, 60, 80, 100, 150, 200, 300]
LAMBDA_VALUES = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]

N_RUNS = 10
MISSING_RATE = 0.5
NOISE_SIGMA = 0.5


def run_sensitivity_anchors(dataset_name: str, data_dir: str | None = None):
    views, labels = load_dataset(dataset_name, data_dir)
    c = N_CLUSTERS[dataset_name]
    n = views[0].shape[0]
    V = len(views)

    print(f"\n  Anchor sensitivity (lambda=1.0, joint degradation)")
    print(f"  {'m':>6} {'ACC':>8} {'NMI':>8} {'ARI':>8}")
    print(f"  {'-'*34}")

    for m in ANCHOR_VALUES:
        if m >= n // 2:
            continue
        all_results = []
        for run in range(N_RUNS):
            rng = np.random.default_rng(run)
            noisy_views = add_gaussian_noise(views, NOISE_SIGMA, rng)
            M = generate_missing_mask(n, V, MISSING_RATE, rng)

            model = QASEF(
                n_clusters=c, n_anchors=m, k_anchors=5,
                lam=1.0, max_iter=100, tol=1e-5,
                random_state=run, verbose=False,
            )
            pred = model.fit_predict(noisy_views, M)
            metrics = evaluate_all(labels, pred)
            all_results.append(metrics)

        acc = np.mean([r["ACC"] for r in all_results])
        nmi_val = np.mean([r["NMI"] for r in all_results])
        ari_val = np.mean([r["ARI"] for r in all_results])
        print(f"  {m:>6} {acc:>8.4f} {nmi_val:>8.4f} {ari_val:>8.4f}")


def run_sensitivity_lambda(dataset_name: str, data_dir: str | None = None):
    views, labels = load_dataset(dataset_name, data_dir)
    c = N_CLUSTERS[dataset_name]
    n = views[0].shape[0]
    V = len(views)
    m = min(c + 50, n // 2)

    print(f"\n  Lambda sensitivity (m={m}, joint degradation)")
    print(f"  {'lambda':>8} {'ACC':>8} {'NMI':>8} {'ARI':>8}")
    print(f"  {'-'*36}")

    for lam in LAMBDA_VALUES:
        all_results = []
        for run in range(N_RUNS):
            rng = np.random.default_rng(run)
            noisy_views = add_gaussian_noise(views, NOISE_SIGMA, rng)
            M = generate_missing_mask(n, V, MISSING_RATE, rng)

            model = QASEF(
                n_clusters=c, n_anchors=m, k_anchors=5,
                lam=lam, max_iter=100, tol=1e-5,
                random_state=run, verbose=False,
            )
            pred = model.fit_predict(noisy_views, M)
            metrics = evaluate_all(labels, pred)
            all_results.append(metrics)

        acc = np.mean([r["ACC"] for r in all_results])
        nmi_val = np.mean([r["NMI"] for r in all_results])
        ari_val = np.mean([r["ARI"] for r in all_results])
        print(f"  {lam:>8.3f} {acc:>8.4f} {nmi_val:>8.4f} {ari_val:>8.4f}")


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
            run_sensitivity_anchors(ds, data_dir)
            run_sensitivity_lambda(ds, data_dir)
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
