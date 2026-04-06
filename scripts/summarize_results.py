#!/usr/bin/env python
"""Aggregate per-run results into CSV / JSON summary tables.

Example
-------
python scripts/summarize_results.py --results_dir results/random_pruning
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Summarize pruning experiment results")
    p.add_argument("--results_dir", type=str, default="results/random_pruning")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write summaries (default: <results_dir>/summary)")
    return p.parse_args()


def collect_runs(results_dir: str):
    """Walk <dataset>/<prune_X>/<seed_Y>/results.json and return list of dicts."""
    rd = Path(results_dir)
    runs = []
    for ds_dir in sorted(rd.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name in ("summary",):
            continue
        for pr_dir in sorted(ds_dir.iterdir()):
            if not pr_dir.is_dir():
                continue
            for seed_dir in sorted(pr_dir.iterdir()):
                rf = seed_dir / "results.json"
                if not rf.exists():
                    continue
                r = json.loads(rf.read_text())
                runs.append({
                    "dataset": r.get("dataset", ds_dir.name),
                    "pruning_ratio": r.get("pruning_ratio"),
                    "seed": r.get("seed"),
                    "retained_size": r.get("retained_size"),
                    "pruned_size": r.get("pruned_size"),
                    "test_acc": r.get("final_test_acc"),
                    "best_test_acc": r.get("best_test_acc"),
                    "train_acc": r.get("final_train_acc"),
                    "elapsed_seconds": r.get("elapsed_seconds"),
                    "output_dir": str(seed_dir),
                })
    return runs


def aggregate(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["pruning_ratio"])].append(r)

    agg = []
    for (ds, pr), grp in sorted(groups.items()):
        accs = [r["test_acc"] for r in grp]
        bests = [r["best_test_acc"] for r in grp]
        agg.append({
            "dataset": ds,
            "pruning_ratio": pr,
            "retained_size": grp[0]["retained_size"],
            "pruned_size": grp[0]["pruned_size"],
            "num_runs": len(grp),
            "mean_test_acc": float(np.mean(accs)),
            "var_test_acc": float(np.var(accs)),
            "std_test_acc": float(np.std(accs)),
            "mean_best_test_acc": float(np.mean(bests)),
            "var_best_test_acc": float(np.var(bests)),
            "std_best_test_acc": float(np.std(bests)),
            "per_run_test_acc": accs,
            "per_run_best_test_acc": bests,
        })
    return agg


def write_csv(rows, fields, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    out = Path(args.output_dir or args.results_dir) / "summary"
    out.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(args.results_dir)
    if not runs:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found {len(runs)} runs")

    # per-run CSV
    run_fields = [
        "dataset", "pruning_ratio", "seed", "retained_size", "pruned_size",
        "test_acc", "best_test_acc", "train_acc", "elapsed_seconds", "output_dir",
    ]
    run_csv = out / "all_runs.csv"
    write_csv(runs, run_fields, run_csv)
    print(f"Per-run CSV  -> {run_csv}")

    # aggregated CSV + JSON
    agged = aggregate(runs)
    agg_fields = [
        "dataset", "pruning_ratio", "retained_size", "pruned_size", "num_runs",
        "mean_test_acc", "var_test_acc", "std_test_acc",
        "mean_best_test_acc", "var_best_test_acc", "std_best_test_acc",
    ]
    agg_csv = out / "aggregated.csv"
    write_csv(agged, agg_fields, agg_csv)
    print(f"Aggregated CSV  -> {agg_csv}")

    agg_json = out / "aggregated.json"
    with open(agg_json, "w") as f:
        json.dump(agged, f, indent=2)
    print(f"Aggregated JSON -> {agg_json}")

    # print table
    hdr = (
        f"{'Dataset':<15} {'Prune%':>7} {'Retained':>10} "
        f"{'Runs':>5} {'MeanAcc':>9} {'Std':>7} {'MeanBest':>9}"
    )
    print(f"\n{'=' * len(hdr)}")
    print(hdr)
    print(f"{'-' * len(hdr)}")
    for a in agged:
        print(
            f"{a['dataset']:<15} {a['pruning_ratio']:>7.0f} "
            f"{a['retained_size']:>10} {a['num_runs']:>5} "
            f"{a['mean_test_acc']:>8.2f}% {a['std_test_acc']:>6.2f} "
            f"{a['mean_best_test_acc']:>8.2f}%"
        )
    print(f"{'=' * len(hdr)}")


if __name__ == "__main__":
    main()
