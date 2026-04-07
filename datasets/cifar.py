"""CIFAR-10 and CIFAR-100 loaders with standard and strong augmentation."""

from typing import Tuple

import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _cifar_transform(
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    train: bool,
) -> T.Compose:
    """Standard CIFAR augmentation: RandomCrop + HorizontalFlip."""
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def _cifar100_strong_transform(
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    train: bool,
) -> T.Compose:
    """Strong augmentation for CIFAR-100:
    Resize → ColorJitter → RandomHorizontalFlip → RandomRotation (±15°)
    → RandomAffine → GaussianBlur → RandomErasing
    (Mixup/CutMix and LabelSmoothing are applied in training loop)
    """
    if train:
        return T.Compose([
            T.Resize(32),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def get_cifar10(root: str, train: bool = True, download: bool = False) -> Dataset:
    transform = _cifar_transform(CIFAR10_MEAN, CIFAR10_STD, train=train)
    return torchvision.datasets.CIFAR10(
        root=root, train=train, download=download, transform=transform,
    )


def get_cifar100(
    root: str,
    train: bool = True,
    download: bool = False,
    strong_aug: bool = False,
) -> Dataset:
    """Get CIFAR-100 dataset.

    Args:
        strong_aug: If True, use strong augmentation (ColorJitter, Rotation,
                    Affine, GaussianBlur, RandomErasing). Mixup/CutMix are
                    applied separately in the training loop.
    """
    if strong_aug and train:
        transform = _cifar100_strong_transform(CIFAR100_MEAN, CIFAR100_STD, train=train)
    else:
        transform = _cifar_transform(CIFAR100_MEAN, CIFAR100_STD, train=train)
    return torchvision.datasets.CIFAR100(
        root=root, train=train, download=download, transform=transform,
    )
