# Dataset Pruning – Reproducible Experiment Pipeline

Reproduce the dataset-pruning baselines from  
*"Dataset Pruning: Reducing Training Data by Examining Generalization"*.

Currently implemented: **Random Pruning**.  
The pipeline is designed so that additional methods (Herding, Forgetting, GraNd, EL2N, Influence-score, Data Pruning, Data Pruning++) can be added by subclassing `pruning.base.PruningStrategy` on their own experiment branches.

---

## Project Structure

```
configs/
  random_pruning.yaml          # hyper-parameters & pruning ratios
datasets/
  __init__.py                  # build_datasets() factory
  cifar.py                     # CIFAR-10 / CIFAR-100 loaders
  tinyimagenet.py              # Tiny ImageNet loader (auto-reorganises val/)
pruning/
  base.py                      # PruningStrategy ABC
  random_pruning.py            # RandomPruningStrategy
engine/
  trainer.py                   # train_and_evaluate(), build_model()
  evaluator.py                 # evaluate()
  utils.py                     # set_seed(), setup_logger(), AverageMeter, I/O
scripts/
  run_random_pruning.py        # main experiment runner
  summarize_results.py         # aggregate per-run results → CSV / JSON
  plot_random_pruning.py       # Figure-2-style accuracy-vs-pruning plot
docs/
  EXPERIMENT_PLAN.md           # overall experiment plan & branch strategy
```

---

## Dataset Paths

| Dataset | Expected location | Notes |
|---------|-------------------|-------|
| CIFAR-10 | `./data/cifar-10-batches-py/` | `torchvision` auto-downloads with `--download` |
| CIFAR-100 | `./data/cifar-100-python/` | same |
| Tiny ImageNet | `./data/tiny-imagenet-200/` | manual download from [cs231n.stanford.edu](http://cs231n.stanford.edu/tiny-imagenet-200.zip) |

Modify paths in `configs/random_pruning.yaml` if your data lives elsewhere.

---

## Requirements

```
torch >= 1.10
torchvision >= 0.11
numpy
pyyaml
matplotlib          # only for plotting
```

---

## Quick Start

### 1. Run one dataset (for debugging)

```bash
python scripts/run_random_pruning.py \
    --config configs/random_pruning.yaml \
    --datasets cifar10 \
    --seeds 0 1 2 \
    --device cuda
```

### 2. Run all three datasets

```bash
python scripts/run_random_pruning.py \
    --config configs/random_pruning.yaml \
    --datasets cifar10 cifar100 tinyimagenet \
    --seeds 0 1 2 \
    --device cuda
```

### 3. Resume interrupted experiments

```bash
python scripts/run_random_pruning.py \
    --config configs/random_pruning.yaml \
    --resume \
    --device cuda
```

Runs whose `results.json` already exists will be skipped.

### 4. Aggregate results

```bash
python scripts/summarize_results.py --results_dir results/random_pruning
```

Produces:
- `results/random_pruning/summary/all_runs.csv` — one row per run
- `results/random_pruning/summary/aggregated.csv` — one row per (dataset, pruning_ratio)
- `results/random_pruning/summary/aggregated.json`

### 5. Generate plots

```bash
python scripts/plot_random_pruning.py --results_dir results/random_pruning
```

Saves `random_pruning_results.png` and `.pdf` in `results/random_pruning/summary/`.

---

## Output Directory Layout

```
results/random_pruning/
  cifar10/
    prune_0/
      seed_0/
        config.json               # full run config snapshot
        retained_indices.npy      # exact indices used for training
        results.json              # final & best accuracy, timing
        training_history.json     # per-epoch metrics
        training.log              # human-readable log
      seed_1/
      seed_2/
    prune_2/
    ...
  cifar100/
  tinyimagenet/
  all_results.json
  summary/
    all_runs.csv
    aggregated.csv
    aggregated.json
    random_pruning_results.png
    random_pruning_results.pdf
```

---

## Training Protocol (matches paper)

| Setting | Value |
|---------|-------|
| Architecture | CIFAR-style ResNet-50 (`models/resnet.py`: 3×3 stem, Bottleneck `[3,4,6,3]`, `2048→50→C`) |
| Epochs | 200 |
| Batch size | 256 |
| Optimizer | SGD (momentum 0.9, weight decay 5e-4) |
| Learning rate | 0.1 → cosine annealing |
| Data augmentation | RandomCrop + RandomHorizontalFlip |
| Evaluation metric | Top-1 test accuracy (%) |
| Repetitions | 3 seeds per (dataset, pruning_ratio) |

Each run trains a **freshly initialised** ResNet-50 (project implementation, not `torchvision.models`) on the pruned subset; no fine-tuning from a full-data checkpoint.

---

## Adding a New Pruning Method

1. Create a new branch: `git checkout -b exp/baseline-<name> main`
2. Add `pruning/<name>.py` implementing `PruningStrategy.select_indices()`
3. Wire it into a new runner script or extend `run_random_pruning.py`
4. Re-use the same summarise / plot scripts
