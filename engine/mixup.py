"""Mixup and CutMix data augmentation for training."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.8):
    """Apply Mixup augmentation.

    Args:
        x: Input images (B, C, H, W).
        y: Target labels (B,).
        alpha: Beta distribution parameter.

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.8):
    """Apply CutMix augmentation.

    Args:
        x: Input images (B, C, H, W).
        y: Target labels (B,).
        alpha: Beta distribution parameter.

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to actual area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixed loss for Mixup/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MixupCutmix:
    """Wrapper to randomly apply Mixup or CutMix with given probability."""

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 0.8,
        prob: float = 0.5,
        switch_prob: float = 0.5,
        enabled: bool = True,
    ):
        """
        Args:
            mixup_alpha: Mixup beta distribution alpha.
            cutmix_alpha: CutMix beta distribution alpha.
            prob: Probability of applying either Mixup or CutMix.
            switch_prob: Probability of choosing Mixup over CutMix when applied.
            enabled: Master switch to enable/disable.
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.enabled = enabled

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """Apply Mixup or CutMix to batch.

        Returns:
            x, y_a, y_b, lam, use_mix (bool)
        """
        if not self.enabled or np.random.rand() > self.prob:
            return x, y, y, 1.0, False

        if np.random.rand() < self.switch_prob:
            # Mixup
            return (*mixup_data(x, y, self.mixup_alpha), True)
        else:
            # CutMix
            return (*cutmix_data(x, y, self.cutmix_alpha), True)
