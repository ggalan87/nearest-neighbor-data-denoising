import copy
import os
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Callable, Dict, Any
import torchvision.transforms.functional as TF

from torchvision.datasets.cifar import CIFAR10

from lightning.data.dataset_utils import pack_decoupled_data
from lightning.data.datasets.base import RecordsDatasetWithNoise


class NoisyCIFAR10(RecordsDatasetWithNoise, CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 labels_noise_perc: Dict[str, float] = MappingProxyType({'symmetric': 0.5}),
                 use_cached_dataset: bool = True
                 ):

        RecordsDatasetWithNoise.__init__(self)
        CIFAR10.__init__(self, root=root, train=train, transform=transform, target_transform=target_transform,
                         download=download)

        # Assign extra arguments
        self._use_cached_dataset = use_cached_dataset
        self._labels_noise_perc = labels_noise_perc

        # Initialize the path of the file which contains the noisy targets
        self.initialize_noisy_targets_filepath()

        # The call below raises exceptions if some criteria are not met
        self.check_args()

        #
        self.convert_to_records()

        # Load noisy data after conversion to records
        self.initialize_noisy_data(train)

    def check_args(self):
        if self._use_cached_dataset and not self._noisy_targets_filepath.exists():
            raise AssertionError(
                f'Required cached dataset, however default path {self._noisy_targets_filepath} does not exist'
            )

        if not self._use_cached_dataset and self._noisy_targets_filepath.exists():
            raise AssertionError(
                f'Required creation of noisy targets, however {self._noisy_targets_filepath} already exists. '
                f'Please delete it manually if you really want to.'
            )

    @property
    def filename_tag(self):
        symmetric_noise = self._labels_noise_perc['symmetric']
        return f'symmetric_noise_{symmetric_noise}'


def pass_dataset():
    dataset = NoisyCIFAR10(root='/data/datasets', labels_noise_perc={'symmetric': 0.1}, use_cached_dataset=False)
    pass


if __name__ == '__main__':
    pass_dataset()
