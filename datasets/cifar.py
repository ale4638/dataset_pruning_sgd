"""CIFAR-10 and CIFAR-100 loaders with standard augmentation."""

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


def get_cifar10(root: str, train: bool = True, download: bool = False) -> Dataset:
    transform = _cifar_transform(CIFAR10_MEAN, CIFAR10_STD, train=train)
    return torchvision.datasets.CIFAR10(
        root=root, train=train, download=download, transform=transform,
    )


def get_cifar100(root: str, train: bool = True, download: bool = False) -> Dataset:
    transform = _cifar_transform(CIFAR100_MEAN, CIFAR100_STD, train=train)
    return torchvision.datasets.CIFAR100(
        root=root, train=train, download=download, transform=transform,
    )
