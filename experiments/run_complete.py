"""Experiment: Complete data setting (Table 2 in the paper).

Evaluates QASEF on all 12 datasets with fully available views.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qasef import QASEF
from qasef.metrics import evaluate_all
from data.loader import load_dataset

DATASETS = [
    "ORL", "100leaves", "Handwritten", "Caltech101-20", "BDGP",
    "Scene-15", "Caltech101", "Reuters", "NUS", "Cifar100",
    "MNIST", "Caltech256",
]

# Dataset-specific cluster counts
N_CLUSTERS = {
    "ORL": 40, "100leaves": 100, "Handwritten": 10, "Caltech101-20": 20,
    "BDGP": 5, "Scene-15": 15, "Caltech101": 101, "Reuters": 6,
    "NUS": 31, "Cifar100": 100, "MNIST": 10, "Caltech256": 256,
}

N_RUNS = 10


def run_complete(dataset_name: str, data_dir: str | None = None):
    views, labels = load_dataset(dataset_name, data_dir)
    c = N_CLUSTERS[dataset_name]
    n = views[0].shape[0]
    m = min(c + 50, n // 2)

    all_results = []
    for run in range(N_RUNS):
        model = QASEF(
            n_clusters=c, n_anchors=m, k_anchors=5,
            lam=1.0, max_iter=100, tol=1e-5,
            random_state=run, verbose=False,
        )
        pred = model.fit_predict(views)
        metrics = evaluate_all(labels, pred)
        all_results.append(metrics)
        print(f"  Run {run + 1:2d}: ACC={metrics['ACC']:.4f}  "
              f"NMI={metrics['NMI']:.4f}  ARI={metrics['ARI']:.4f}")

    # Aggregate
    print(f"\n{'Metric':<12} {'Mean':>8} {'Std':>8}")
    print("-" * 30)
    for metric in ["ACC", "NMI", "Purity", "ARI", "F-score", "Precision"]:
        vals = [r[metric] for r in all_results]
        print(f"{metric:<12} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}")

    return all_results


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
            run_complete(ds, data_dir)
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
