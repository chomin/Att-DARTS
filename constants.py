from enum import Enum, auto, unique
from pathlib import Path

DATA_DIRECTORY = Path('data/')


@unique
class MyDataset(Enum):
    CIFAR10 = auto()
    CIFAR100 = auto()
    ImageNet = auto()
