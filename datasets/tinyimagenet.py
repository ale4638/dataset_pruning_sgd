"""Tiny ImageNet (200 classes, 64×64) loader.

Handles the standard download layout where val images sit in a flat
``val/images/`` folder with labels in ``val/val_annotations.txt``.
On first use the val directory is reorganised into per-class sub-folders
so that ``torchvision.datasets.ImageFolder`` can load it directly.
"""

import os
import shutil
from pathlib import Path

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

TINYIMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINYIMAGENET_STD = (0.2302, 0.2265, 0.2262)


def _tinyimagenet_transform(train: bool) -> T.Compose:
    if train:
        return T.Compose([
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
        ])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
    ])


def _reorganize_val_dir(val_dir: str) -> None:
    """Move val images into per-class sub-directories (idempotent)."""
    val_path = Path(val_dir)
    ann_file = val_path / "val_annotations.txt"
    images_dir = val_path / "images"

    if not ann_file.exists() or not images_dir.exists():
        return  # already reorganised or non-standard layout

    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, class_id = parts[0], parts[1]
            class_dir = val_path / class_id / "images"
            class_dir.mkdir(parents=True, exist_ok=True)
            src = images_dir / fname
            dst = class_dir / fname
            if src.exists():
                shutil.move(str(src), str(dst))

    # Remove the now-empty flat images dir
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()


def get_tinyimagenet(root: str, train: bool = True) -> ImageFolder:
    """Return a TinyImageNet split as an ``ImageFolder`` dataset."""
    root = Path(root)
    if train:
        data_dir = root / "train"
    else:
        data_dir = root / "val"
        _reorganize_val_dir(str(data_dir))

    transform = _tinyimagenet_transform(train=train)
    return ImageFolder(str(data_dir), transform=transform)
