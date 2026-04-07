"""Herding pruning: select samples closest to class cluster centers.

Herding is a greedy algorithm that selects samples such that the mean of
selected samples approximates the mean of all samples in each class.

Reference:
    Welling, M. (2009). Herding dynamical weights to learn.
    ICML 2009.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .base import PruningStrategy

logger = logging.getLogger(__name__)


def extract_features(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
) -> np.ndarray:
    """Extract features from the penultimate layer of the model.

    Args:
        model: Neural network (features extracted before final linear layer).
        dataset: Dataset to extract features from.
        device: Device to run inference on.
        batch_size: Batch size for inference.
        num_workers: DataLoader workers.

    Returns:
        Feature array of shape (N, D).
    """
    model.eval()
    model = model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    features_list = []

    # Hook to capture features before the final linear layer
    features_hook = []

    def hook_fn(module, input, output):
        # input[0] is the feature before linear layer
        features_hook.append(input[0].detach().cpu())

    # Register hook on the last linear layer (linear2 in our CIFAR ResNet)
    # For our model: linear1 -> linear2, we want features before linear1
    # Actually let's hook after adaptive_avg_pool (before linear1)
    # Our model has: ... -> adaptive_avg_pool -> flatten -> linear1 -> linear2
    # We want the flattened features (2048-dim for ResNet50)

    # Find the linear1 layer and hook its input
    hook_handle = None
    for name, module in model.named_modules():
        if name == "linear1":
            hook_handle = module.register_forward_hook(hook_fn)
            break

    if hook_handle is None:
        raise RuntimeError("Could not find 'linear1' layer in model")

    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Extracting features", leave=False):
            inputs = inputs.to(device)
            _ = model(inputs)

    hook_handle.remove()

    features = torch.cat(features_hook, dim=0).numpy()
    logger.info(f"Extracted features: shape={features.shape}")
    return features


def herding_select(
    features: np.ndarray,
    targets: np.ndarray,
    num_select_per_class: dict,
    seed: int = 0,
) -> List[int]:
    """Greedy herding selection per class.

    For each class, iteratively select the sample that minimizes the distance
    between the mean of selected samples and the class mean.

    Args:
        features: Feature array (N, D).
        targets: Label array (N,).
        num_select_per_class: Dict mapping class_id -> number to select.
        seed: Random seed (used for tie-breaking).

    Returns:
        Sorted list of selected indices.
    """
    rng = np.random.RandomState(seed)
    selected_indices = []
    classes = sorted(num_select_per_class.keys())

    for cls in classes:
        cls_mask = targets == cls
        cls_indices = np.where(cls_mask)[0]
        cls_features = features[cls_mask]  # (n_cls, D)
        n_select = num_select_per_class[cls]

        if n_select >= len(cls_indices):
            # Select all
            selected_indices.extend(cls_indices.tolist())
            continue

        if n_select == 0:
            continue

        # Class mean (target)
        cls_mean = cls_features.mean(axis=0)  # (D,)

        # Greedy herding
        selected_local = []  # local indices within cls_features
        selected_sum = np.zeros_like(cls_mean)

        for _ in range(n_select):
            # For each candidate, compute mean if we add it
            # new_mean = (selected_sum + candidate) / (len(selected) + 1)
            # We want to minimize || new_mean - cls_mean ||
            # = || (selected_sum + candidate) / (k+1) - cls_mean ||
            # = || selected_sum + candidate - (k+1) * cls_mean || / (k+1)
            # Minimizing this is equivalent to minimizing:
            # || selected_sum + candidate - (k+1) * cls_mean ||^2

            k = len(selected_local)
            target_sum = (k + 1) * cls_mean  # What we want selected_sum + x to be close to
            residual = target_sum - selected_sum  # (D,)

            # For each candidate x, we want to minimize || x - residual ||^2
            # i.e., select x closest to residual
            candidates_mask = np.ones(len(cls_features), dtype=bool)
            candidates_mask[selected_local] = False
            candidate_indices = np.where(candidates_mask)[0]

            if len(candidate_indices) == 0:
                break

            candidate_features = cls_features[candidate_indices]  # (n_cand, D)
            distances = np.linalg.norm(candidate_features - residual, axis=1)

            # Find minimum (with random tie-breaking)
            min_dist = distances.min()
            min_indices = np.where(distances == min_dist)[0]
            chosen_local_idx = candidate_indices[rng.choice(min_indices)]

            selected_local.append(chosen_local_idx)
            selected_sum += cls_features[chosen_local_idx]

        # Convert local indices to global indices
        for local_idx in selected_local:
            selected_indices.append(cls_indices[local_idx])

    return sorted(selected_indices)


class HerdingPruningStrategy(PruningStrategy):
    """Herding-based dataset pruning.

    Selects samples closest to class cluster centers using greedy herding.
    Requires a model to extract features.
    """

    def __init__(
        self,
        model: nn.Module = None,
        device: torch.device = None,
        batch_size: int = 256,
        num_workers: int = 4,
    ):
        """
        Args:
            model: Model for feature extraction. If None, must be set later via
                   `set_model()` or passed to `select_indices()`.
            device: Device for inference.
            batch_size: Batch size for feature extraction.
            num_workers: DataLoader workers.
        """
        self.model = model
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._features_cache = None
        self._targets_cache = None

    def set_model(self, model: nn.Module, device: torch.device = None):
        """Set the model for feature extraction."""
        self.model = model
        if device is not None:
            self.device = device
        self._features_cache = None  # Clear cache when model changes

    def select_indices(
        self,
        dataset_size: int,
        pruning_ratio: float,
        seed: int,
        targets: Optional[np.ndarray] = None,
        num_classes: Optional[int] = None,
        dataset: Optional[Dataset] = None,
        **kwargs,
    ) -> List[int]:
        """Select indices using herding.

        Args:
            dataset_size: Total number of training examples.
            pruning_ratio: Percentage of data to remove (0-100).
            seed: Random seed for tie-breaking.
            targets: Class label array (required).
            num_classes: Number of classes (required).
            dataset: The dataset object (required for feature extraction).

        Returns:
            Sorted list of retained sample indices.
        """
        if targets is None:
            raise ValueError("Herding requires targets (class labels)")
        if num_classes is None:
            raise ValueError("Herding requires num_classes")
        if dataset is None:
            raise ValueError("Herding requires dataset for feature extraction")
        if self.model is None:
            raise ValueError("Herding requires a model. Call set_model() first.")

        retained_size = self.compute_retained_size(dataset_size, pruning_ratio)

        if retained_size >= dataset_size:
            indices = list(range(dataset_size))
            logger.info(
                "Herding | ratio=%.1f%% | seed=%d | total=%d | retained=%d (all)",
                pruning_ratio, seed, dataset_size, len(indices),
            )
            return indices

        # Extract features (cache for efficiency when running multiple seeds)
        if self._features_cache is None or len(self._features_cache) != len(dataset):
            logger.info("Extracting features for herding...")
            self._features_cache = extract_features(
                self.model, dataset, self.device, self.batch_size, self.num_workers,
            )
            self._targets_cache = targets

        # Compute per-class selection count (proportional to class size)
        unique_classes, class_counts = np.unique(targets, return_counts=True)
        total_samples = len(targets)

        # Distribute retained_size proportionally across classes
        num_select_per_class = {}
        remaining = retained_size
        for cls, count in zip(unique_classes, class_counts):
            # Proportional allocation
            n_select = int(round(count / total_samples * retained_size))
            n_select = min(n_select, count)  # Can't select more than available
            num_select_per_class[cls] = n_select
            remaining -= n_select

        # Distribute any remaining due to rounding
        if remaining > 0:
            for cls in unique_classes:
                if remaining <= 0:
                    break
                max_add = class_counts[cls] - num_select_per_class[cls]
                add = min(remaining, max_add)
                num_select_per_class[cls] += add
                remaining -= add

        # Greedy herding selection
        indices = herding_select(
            self._features_cache,
            targets,
            num_select_per_class,
            seed=seed,
        )

        logger.info(
            "Herding | ratio=%.1f%% | seed=%d | total=%d | retained=%d | pruned=%d",
            pruning_ratio, seed, dataset_size, len(indices), dataset_size - len(indices),
        )

        return indices
