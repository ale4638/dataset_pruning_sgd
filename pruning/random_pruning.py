"""Random pruning: uniformly sample a subset without replacement."""

import logging
from typing import List, Optional

import numpy as np

from .base import PruningStrategy

logger = logging.getLogger(__name__)


class RandomPruningStrategy(PruningStrategy):
    """Plain global random pruning (no class balancing by default)."""

    def select_indices(
        self,
        dataset_size: int,
        pruning_ratio: float,
        seed: int,
        targets: Optional[np.ndarray] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ) -> List[int]:
        retained_size = self.compute_retained_size(dataset_size, pruning_ratio)

        if retained_size >= dataset_size:
            indices = list(range(dataset_size))
        else:
            rng = np.random.RandomState(seed)
            indices = sorted(
                rng.choice(dataset_size, size=retained_size, replace=False).tolist()
            )

        logger.info(
            "RandomPruning | ratio=%.1f%% | seed=%d | "
            "total=%d | retained=%d | pruned=%d",
            pruning_ratio, seed, dataset_size,
            len(indices), dataset_size - len(indices),
        )
        return indices
