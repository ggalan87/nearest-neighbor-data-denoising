import os
from typing import Optional, Callable

from PIL import Image
import torchvision.datasets
import torchvision.transforms.functional as TF

from lightning.data.datasets.base import RecordsDataset


class FashionMNIST(RecordsDataset, torchvision.datasets.FashionMNIST):
    """
    An extension to FashionMNIST for exposing additional data apart from standard transformed image and groundtruth label. By
    specifying DatasetExt as the first inherited class we specify that implementation of this class should be used
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False, ):
        super().__init__(root, train, transform, target_transform, download)

        self.convert_to_records()

    def get_pil_image(self, img) -> Image:
        return TF.to_pil_image(img, mode='L')
