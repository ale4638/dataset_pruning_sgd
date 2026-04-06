"""Model evaluation on a test / validation set."""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import AverageMeter


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Evaluate model and return loss / top-1 accuracy.

    Returns:
        dict with keys: loss, accuracy, correct, total
    """
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return {
        "loss": losses.avg,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
