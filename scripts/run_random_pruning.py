#!/usr/bin/env python
"""Run random pruning experiments across datasets, pruning ratios, and seeds.

Example usage
-------------
# All three datasets, default 3 seeds:
python scripts/run_random_pruning.py --config configs/random_pruning.yaml

# Single dataset for debugging:
python scripts/run_random_pruning.py --config configs/random_pruning.yaml \
    --datasets cifar10 --seeds 0 --device cuda
"""

import argparse
import json
import os
import sys

# Force unbuffered stdout/stderr so output appears immediately even under
# `conda run` or piped execution.
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, "reconfigure") else None

from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("[init] importing libraries...", flush=True)
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from datasets import build_datasets
from engine.trainer import train_and_evaluate
from engine.utils import save_json, seed_worker, set_seed, setup_logger
from pruning import RandomPruningStrategy
print("[init] imports done.", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_targets(dataset) -> np.ndarray:
    """Extract integer label array from a PyTorch dataset."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    if hasattr(dataset, "samples"):
        return np.array([s[1] for s in dataset.samples])
    raise ValueError("Cannot extract targets from dataset")


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single(
    dataset_name: str,
    pruning_ratio: float,
    seed: int,
    train_dataset,
    test_dataset,
    num_classes: int,
    training_config: dict,
    device: torch.device,
    output_root: str,
    save_checkpoints: bool,
    deterministic: bool,
    resume: bool,
    augmentation_config: dict = None,
) -> dict:
    """Select subset → train from scratch → evaluate → save results."""

    output_dir = (
        Path(output_root) / dataset_name
        / f"prune_{int(pruning_ratio)}" / f"seed_{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already finished
    results_file = output_dir / "results.json"
    if resume and results_file.exists():
        print(f"  [Skip] {output_dir}  (results.json exists)")
        return json.loads(results_file.read_text())

    log_name = f"{dataset_name}_p{int(pruning_ratio)}_s{seed}"
    lgr = setup_logger(log_name, str(output_dir / "training.log"))

    # --- subset selection (uses its own RNG) ---
    strategy = RandomPruningStrategy()
    targets = get_targets(train_dataset)
    indices = strategy.select_indices(
        dataset_size=len(train_dataset),
        pruning_ratio=pruning_ratio,
        seed=seed,
        targets=targets,
        num_classes=num_classes,
    )
    np.save(str(output_dir / "retained_indices.npy"), np.array(indices))

    # --- dataloaders ---
    set_seed(seed, deterministic=deterministic)
    g = torch.Generator()
    g.manual_seed(seed)

    subset = Subset(train_dataset, indices)
    batch_size = training_config["batch_size"]
    num_workers = training_config.get("num_workers", 4)
    pin = device.type == "cuda"

    train_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    # --- save run config ---
    run_cfg = {
        "dataset": dataset_name,
        "pruning_ratio": pruning_ratio,
        "seed": seed,
        "original_size": len(train_dataset),
        "retained_size": len(indices),
        "pruned_size": len(train_dataset) - len(indices),
        "num_classes": num_classes,
        "training": training_config,
        "deterministic": deterministic,
        "device": str(device),
        "augmentation": augmentation_config,
    }
    save_json(run_cfg, str(output_dir / "config.json"))

    print(
        f"[run] {dataset_name} | prune={pruning_ratio}% | seed={seed} | "
        f"train {len(train_dataset)}->{len(indices)} | test {len(test_dataset)}",
        flush=True,
    )

    lgr.info("=" * 60)
    lgr.info("Dataset: %s | Prune: %.1f%% | Seed: %d", dataset_name, pruning_ratio, seed)
    lgr.info(
        "Train: %d -> %d samples | Test: %d samples",
        len(train_dataset), len(indices), len(test_dataset),
    )
    lgr.info("=" * 60)

    print("[run] building model & starting training...", flush=True)
    # --- train & eval (seed set again so model init is reproducible) ---
    set_seed(seed, deterministic=deterministic)

    results = train_and_evaluate(
        num_classes=num_classes,
        train_loader=train_loader,
        test_loader=test_loader,
        config=training_config,
        device=device,
        output_dir=str(output_dir),
        save_checkpoints=save_checkpoints,
        run_logger=lgr,
        dataset_name=dataset_name,
        pruning_ratio=pruning_ratio,
        seed=seed,
        augmentation_config=augmentation_config,
    )

    results.update({
        "dataset": dataset_name,
        "pruning_ratio": pruning_ratio,
        "seed": seed,
        "retained_size": len(indices),
        "pruned_size": len(train_dataset) - len(indices),
    })
    save_json(results, str(results_file))

    # cleanup logger handlers to avoid fd leaks
    for h in lgr.handlers[:]:
        h.close()
        lgr.removeHandler(h)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Random Pruning Experiments")
    p.add_argument("--config", type=str, default="configs/random_pruning.yaml",
                   help="Path to YAML config file")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Dataset(s) to run (default: all in config)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds (default: from config)")
    p.add_argument("--device", type=str, default="auto",
                   help="cuda / cpu / auto")
    p.add_argument("--save-checkpoints", action="store_true",
                   help="Save best & final model checkpoints (large disk usage)")
    p.add_argument("--resume", action="store_true",
                   help="Skip runs whose results.json already exists")
    p.add_argument("--download", action="store_true",
                   help="Let torchvision download missing CIFAR data")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[main] loading config: {args.config}", flush=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[main] Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[main] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    training_cfg = cfg["training"]
    ds_cfgs = cfg["datasets"]
    seeds = args.seeds or cfg["pruning"]["seeds"]
    deterministic = cfg["pruning"].get("deterministic", True)
    output_root = cfg["output"]["root"]
    save_ckpt = args.save_checkpoints or cfg["output"].get("save_checkpoints", False)

    dataset_names = args.datasets or list(ds_cfgs.keys())
    all_results = []

    for ds_name in dataset_names:
        ds_cfg = ds_cfgs[ds_name]
        print(f"\n{'=' * 60}", flush=True)
        print(f"[data] Loading dataset: {ds_name} from {ds_cfg['root']}", flush=True)
        print(f"{'=' * 60}", flush=True)

        # Check for strong augmentation config (e.g., for CIFAR-100)
        aug_cfg = ds_cfg.get("augmentation", {})
        strong_aug = aug_cfg.get("strong_aug", False)

        train_ds, test_ds, num_classes = build_datasets(
            ds_name, ds_cfg["root"], download=args.download, strong_aug=strong_aug,
        )
        print(
            f"[data] {ds_name}: Train={len(train_ds)}  Test={len(test_ds)}  "
            f"Classes={num_classes}", flush=True,
        )
        if strong_aug:
            print(
                f"[data] Strong augmentation ENABLED: "
                f"mixup_alpha={aug_cfg.get('mixup_alpha', 0)}, "
                f"cutmix_alpha={aug_cfg.get('cutmix_alpha', 0)}, "
                f"label_smoothing={aug_cfg.get('label_smoothing', 0)}",
                flush=True,
            )
        print(
            f"[plan] Pruning ratios: {ds_cfg['pruning_ratios']}  "
            f"Seeds: {seeds}  "
            f"Total runs: {len(ds_cfg['pruning_ratios']) * len(seeds)}",
            flush=True,
        )

        for pr in ds_cfg["pruning_ratios"]:
            for seed in seeds:
                print(
                    f"\n>>> Starting: {ds_name} | prune={pr}% | seed={seed}",
                    flush=True,
                )
                result = run_single(
                    dataset_name=ds_name,
                    pruning_ratio=pr,
                    seed=seed,
                    train_dataset=train_ds,
                    test_dataset=test_ds,
                    num_classes=num_classes,
                    training_config=training_cfg,
                    device=device,
                    output_root=output_root,
                    save_checkpoints=save_ckpt,
                    deterministic=deterministic,
                    resume=args.resume,
                    augmentation_config=aug_cfg if aug_cfg else None,
                )
                all_results.append(result)
                print(
                    f"<<< Done: {ds_name} | prune={pr}% seed={seed} | "
                    f"test_acc={result['final_test_acc']:.2f}% "
                    f"best={result['best_test_acc']:.2f}%",
                    flush=True,
                )

    save_json(all_results, str(Path(output_root) / "all_results.json"))
    print(f"\nAll results saved to {output_root}/")
    print("Run  python scripts/summarize_results.py  to aggregate.")


if __name__ == "__main__":
    main()
