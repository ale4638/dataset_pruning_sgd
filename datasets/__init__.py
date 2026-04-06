"""Dataset factory – returns (train_dataset, test_dataset, num_classes)."""

from typing import Tuple

from torch.utils.data import Dataset

from .cifar import get_cifar10, get_cifar100
from .tinyimagenet import get_tinyimagenet

DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tinyimagenet": 200,
}


def build_datasets(
    name: str,
    root: str,
    download: bool = False,
) -> Tuple[Dataset, Dataset, int]:
    """Build train and test datasets for *name*.

    Returns:
        (train_dataset, test_dataset, num_classes)
    """
    name = name.lower()
    if name == "cifar10":
        train_ds = get_cifar10(root, train=True, download=download)
        test_ds = get_cifar10(root, train=False, download=download)
        return train_ds, test_ds, 10
    elif name == "cifar100":
        train_ds = get_cifar100(root, train=True, download=download)
        test_ds = get_cifar100(root, train=False, download=download)
        return train_ds, test_ds, 100
    elif name == "tinyimagenet":
        train_ds = get_tinyimagenet(root, train=True)
        test_ds = get_tinyimagenet(root, train=False)
        return train_ds, test_ds, 200
    else:
        raise ValueError(f"Unknown dataset: {name}")
