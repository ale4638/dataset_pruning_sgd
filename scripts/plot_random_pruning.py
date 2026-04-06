#!/usr/bin/env python
"""Generate Figure-2-style accuracy-vs-pruning-ratio plots.

Example
-------
python scripts/plot_random_pruning.py --results_dir results/random_pruning
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Plot random pruning results")
    p.add_argument("--results_dir", type=str, default="results/random_pruning")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--metric", type=str, default="mean_best_test_acc",
                   choices=["mean_best_test_acc", "mean_test_acc"],
                   help="Which accuracy to plot")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir or results_dir) / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    agg_file = output_dir / "aggregated.json"
    if not agg_file.exists():
        print(f"aggregated.json not found at {agg_file}")
        print("Run  python scripts/summarize_results.py  first.")
        return

    with open(agg_file) as f:
        aggregated = json.load(f)

    by_dataset: dict = defaultdict(list)
    for entry in aggregated:
        by_dataset[entry["dataset"]].append(entry)

    # deferred matplotlib import so the rest stays lightweight
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DATASET_ORDER = ["cifar10", "cifar100", "tinyimagenet"]
    DATASET_LABELS = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "tinyimagenet": "Tiny ImageNet",
    }
    present = [d for d in DATASET_ORDER if d in by_dataset]
    if not present:
        print("No data to plot.")
        return

    metric_key = args.metric
    std_key = metric_key.replace("mean_", "std_")

    n = len(present)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4.2 * n), squeeze=False)

    for i, ds in enumerate(present):
        ax = axes[i, 0]
        entries = sorted(by_dataset[ds], key=lambda e: e["pruning_ratio"])
        ratios = [e["pruning_ratio"] for e in entries]
        means = [e[metric_key] for e in entries]
        stds = [e[std_key] for e in entries]

        ax.errorbar(
            ratios, means, yerr=stds,
            marker="o", capsize=4, linewidth=2, markersize=6,
            color="royalblue", label="Random Pruning",
        )
        ax.set_xlabel("Pruning Ratio (%)", fontsize=12)
        ax.set_ylabel("Test Acc. (%)", fontsize=12)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ratios)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"random_pruning_results.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
