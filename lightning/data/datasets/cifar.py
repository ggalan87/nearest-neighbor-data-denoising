import os
from typing import Optional, Callable

from PIL import Image
import torchvision.datasets

from lightning.data.datasets.base import RecordsDataset


class CIFAR10(RecordsDataset, torchvision.datasets.CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False, ):

        RecordsDataset.__init__(self)
        torchvision.datasets.CIFAR10.__init__(self, root=root, train=train, transform=transform, target_transform=target_transform,
                         download=download)

        self.convert_to_records()