"""Abstract base class for dataset pruning strategies."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class PruningStrategy(ABC):
    """Interface that every pruning method must implement.

    To add a new method, subclass this and implement ``select_indices``.
    """

    @abstractmethod
    def select_indices(
        self,
        dataset_size: int,
        pruning_ratio: float,
        seed: int,
        targets: Optional[np.ndarray] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ) -> List[int]:
        """Return the sorted list of indices to **retain**.

        Args:
            dataset_size: Total number of training examples.
            pruning_ratio: Percentage of data to *remove* (0–100).
            seed: Random seed for reproducibility.
            targets: Class label array (needed by class-aware methods).
            num_classes: Number of classes (needed by class-aware methods).

        Returns:
            Sorted list of retained sample indices.
        """
        raise NotImplementedError

    @staticmethod
    def compute_retained_size(dataset_size: int, pruning_ratio: float) -> int:
        """Number of samples kept after pruning."""
        return round(dataset_size * (1.0 - pruning_ratio / 100.0))
