"""Model evaluation on a test / validation set."""

import sys
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import AverageMeter


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epoch: int = 0,
    total_epochs: int = 0,
    run_tag: str = "",
) -> Dict[str, float]:
    """Evaluate model and return loss / top-1 accuracy.

    Returns:
        dict with keys: loss, accuracy, correct, total
    """
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(
        dataloader,
        desc=f"{run_tag} Epoch {epoch:>3d}/{total_epochs} [ Test]",
        leave=False,
        bar_format="{l_bar}{bar:30}{r_bar}",
        file=sys.stdout,
    )

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.0 * correct / total

        pbar.set_postfix_str(f"Loss={losses.avg:.4f}  Acc={acc:.2f}%")

    accuracy = 100.0 * correct / total
    return {
        "loss": losses.avg,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
