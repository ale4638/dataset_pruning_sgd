"""Training loop for dataset pruning experiments."""

import logging
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluator import evaluate
from .utils import AverageMeter, save_json

logger = logging.getLogger(__name__)


def build_model(num_classes: int) -> nn.Module:
    """Build a torchvision ResNet50 (random init, no pretrained weights)."""
    try:
        model = models.resnet50(weights=None, num_classes=num_classes)
    except TypeError:
        model = models.resnet50(pretrained=False, num_classes=num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    total_epochs: int = 0,
) -> Dict[str, float]:
    """Run one training epoch and return loss / accuracy."""
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:>3d}/{total_epochs}  [Train]",
        leave=False,
        bar_format="{l_bar}{bar:30}{r_bar}",
        file=sys.stdout,
    )

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.0 * correct / total

        pbar.set_postfix_str(f"Loss={losses.avg:.4f}  Acc={acc:.2f}%")

    accuracy = 100.0 * correct / total
    return {"loss": losses.avg, "accuracy": accuracy}


def train_and_evaluate(
    num_classes: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: str,
    save_checkpoints: bool = False,
) -> dict:
    """Full training + evaluation pipeline.

    Builds a fresh ResNet50, trains for ``config["epochs"]`` epochs with
    SGD + cosine-annealing LR, and evaluates on *test_loader* every epoch.

    Returns:
        dict with final / best metrics and timing.
    """
    os.makedirs(output_dir, exist_ok=True)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"],
    )

    best_test_acc = 0.0
    history: List[dict] = []
    total_epochs = config["epochs"]

    t0 = time.time()
    for epoch in range(1, total_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch=epoch, total_epochs=total_epochs,
        )
        test_metrics = evaluate(
            model, test_loader, device, criterion,
            epoch=epoch, total_epochs=total_epochs,
        )
        scheduler.step()

        is_best = test_metrics["accuracy"] > best_test_acc
        if is_best:
            best_test_acc = test_metrics["accuracy"]

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
            "best_test_acc": best_test_acc,
        }
        history.append(record)

        star = " *" if is_best else ""
        summary = (
            f"Epoch {epoch:>3d}/{total_epochs} │ "
            f"LR {record['lr']:.5f} │ "
            f"Train Loss {train_metrics['loss']:.4f}  Acc {train_metrics['accuracy']:.2f}% │ "
            f"Test Loss {test_metrics['loss']:.4f}  Acc {test_metrics['accuracy']:.2f}% │ "
            f"Best {best_test_acc:.2f}%{star}"
        )
        print(summary, flush=True)

        logger.info(
            "Epoch %3d/%d | LR %.5f | TrainLoss %.4f TrainAcc %.2f%% | "
            "TestLoss %.4f TestAcc %.2f%% | Best %.2f%%",
            epoch, total_epochs, record["lr"],
            train_metrics["loss"], train_metrics["accuracy"],
            test_metrics["loss"], test_metrics["accuracy"],
            best_test_acc,
        )

        if save_checkpoints:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_test_acc": best_test_acc,
            }
            if is_best:
                torch.save(ckpt, os.path.join(output_dir, "checkpoint_best.pth"))
            if epoch == total_epochs:
                torch.save(ckpt, os.path.join(output_dir, "checkpoint_final.pth"))

    elapsed = time.time() - t0
    results = {
        "final_test_acc": history[-1]["test_acc"],
        "best_test_acc": best_test_acc,
        "final_train_acc": history[-1]["train_acc"],
        "final_train_loss": history[-1]["train_loss"],
        "elapsed_seconds": round(elapsed, 1),
        "total_epochs": total_epochs,
    }

    save_json(history, os.path.join(output_dir, "training_history.json"))
    save_json(results, os.path.join(output_dir, "results.json"))

    logger.info(
        "Training complete | %.1f min | Final TestAcc %.2f%% | Best %.2f%%",
        elapsed / 60, results["final_test_acc"], best_test_acc,
    )
    return results
