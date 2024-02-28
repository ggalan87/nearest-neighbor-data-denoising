import os
from typing import Optional, Callable, List

from PIL import Image
import torchvision.datasets
import torchvision.transforms.functional as TF

from lightning.data.datasets.base import DatasetExt, RecordsDataset
from lightning.data.dataset_filter import ListFilter, filter_by


class MNIST(RecordsDataset, torchvision.datasets.MNIST):
    """
    An extension to MNIST for exposing additional data apart from standard transformed image and groundtruth label.
    For this purpose, but also for being used as a subset, it is converted to records dataset, as done with more
    complex datasets.
    """
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)

        self.convert_to_records()

    def get_pil_image(self, img):
        return TF.to_pil_image(img, mode='L')


class MNISTSubset(MNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            included_targets: List[int] = (0, 1, 2),
    ) -> None:
        self._included_targets = included_targets
        super(MNISTSubset, self).__init__(root, train, transform, target_transform, download)

    def convert_to_records(self):
        super().convert_to_records()

        filters = \
            [
                ListFilter('target', list(self._included_targets))
            ]
        self.data = filter_by(self.data, filters)

    @property
    def raw_folder(self) -> str:
        return super().raw_folder.replace('Subset', '')
