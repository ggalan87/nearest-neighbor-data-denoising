import os
import os.path
import copy
import itertools
import pickle
from typing import Tuple, Set, Any, List
from pathlib import Path
from typing import Callable, Optional, Dict
from types import MappingProxyType
from PIL import Image
import torch
from torchvision.datasets import MNIST
import torchvision.transforms.functional as TF
import pandas as pd

from lightning.data.datasets.base import DatasetExt
from lightning.data.dataset_utils import disturb_targets, disturb_records_targets, pack_decoupled_data
from lightning.data.dataset_filter import filter_by, ListFilter

from lightning.ext import logger

# TODO: Reformat this to RecordsDatasetWithNoise and possibly to my MNIST impl (possibly not because they inherit from
#  Record), in order to avoid duplication


class NoisyMNIST(MNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 labels_noise_perc: Dict[int, float] = MappingProxyType({9: 1.0}),
                 use_cached_dataset: bool = True
                 ):
        super().__init__(root, train, transform, target_transform, download)

        # Assign extra arguments
        self._labels_noise_perc = labels_noise_perc
        self._use_cached_dataset = use_cached_dataset

        filename_tag = ','.join(map(lambda kv: f'{kv[0]}_{kv[1]}', labels_noise_perc.items()))
        root_ext = Path(self.dataset_extras_folder)

        # Utilize a predefined filepath which is formed according to the labels
        self._cached_filepath = root_ext / f'noisy_mnist_train_targets_opt-{filename_tag}.pkl'

        # The call below raises exceptions if some criteria are not met
        self.check_args()

        self.convert_to_records()

        if train:
            if self._use_cached_dataset:
                logger.info(f'Loading from {self._cached_filepath}')
                self.load_noisy_data()
            else:
                logger.info(f'Creating MNIST noisy targets {labels_noise_perc}')
                self.data = disturb_records_targets(self.data, self._labels_noise_perc, key_to_disturb='target')

                logger.info(f'Saving targets and indices {self._cached_filepath}')
                self.save_noisy_data()
        else:
            self.set_test_noisy_data()

    def check_args(self):
        if self.train and self._labels_noise_perc is None or len(self._labels_noise_perc) < 1:
            raise AssertionError(
                'At least one element is needed for the parameter labels_noise_perc,'
                'else dataset degenerates to simple MNIST')

        if self._use_cached_dataset and not self._cached_filepath.exists():
            raise AssertionError(
                f'Required cached dataset, however default pa   th {self._cached_filepath} does not exist'
            )

    def convert_to_records(self):
        # Transform data into list of records (dictionaries)
        self.data = pack_decoupled_data(image=self.data, target=self.targets)

    def save_noisy_data(self):
        data_df = pd.DataFrame(self.data)
        targets = data_df['target']
        noisy_indices = data_df['is_noisy']
        noisy_data = \
            {
                'targets': targets,
                'noisy_indices': noisy_indices
            }

        with open(self._cached_filepath, 'wb') as f:
            pickle.dump(noisy_data, f)

    def set_test_noisy_data(self):
        data_df = pd.DataFrame(self.data)

        # None of the data are noisy
        noisy_indices = torch.zeros((len(self.data),), dtype=torch.bool)
        data_df['is_noisy'] = noisy_indices.numpy()

        self.data = data_df.to_dict('records')

    def load_noisy_data(self):
        with open(self._cached_filepath, 'rb') as f:
            noisy_data = pickle.load(f)

        data_df = pd.DataFrame(self.data)

        # Create an entry to keep the original targets
        data_df['target_orig'] = data_df.loc[:, 'target']

        # Override with noisy targets
        data_df['target'] = noisy_data['targets']
        data_df['is_noisy'] = noisy_data['noisy_indices']

        self.data = data_df.to_dict('records')

        # Delete targets member since all required data are in the record now
        del self.targets

    @property
    def raw_folder(self) -> str:
        """
        Override the default implementation which takes class name as output path, because the dataset is exactly
        the same as the original.
        """
        return os.path.join(self.root, self.__class__.__name__.replace('Noisy', ''), 'raw')

    @property
    def dataset_extras_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__.replace('Ext', ''))

    def __getitem__(self, index: int) -> Any:
        data_entry = copy.copy(self.data[index])

        image = TF.to_pil_image(data_entry['image'], mode='L')

        if self.transform is not None:
            image = self.transform(image)

        data_entry['image'] = image
        data_entry['data_idx'] = index

        return data_entry


class NoisyMNISTSubset(NoisyMNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 labels_noise_perc: Dict[int, float] = MappingProxyType({9: 1.0}),
                 use_cached_dataset: bool = True,
                 included_targets: List[int] = (0, 1, 2),
                 ):
        self._included_targets = included_targets
        super().__init__(root, train, transform, target_transform, download, labels_noise_perc, use_cached_dataset)

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


def obtain_noisy_label_configs(n_classes: int, n_excluded: Tuple, noisiness_perc: float = 1.0):
    configs = []
    for n in n_excluded:
        noisy_labels_combos = list(itertools.combinations(range(n_classes), n))
        configs.extend([{l: noisiness_perc for l in combo} for combo in noisy_labels_combos])
    return configs


def save_noisy_mnist_configs():
    # noisy_label_configs = obtain_noisy_label_configs(n_classes=10, n_excluded=(1, 2), noisiness_perc=1.0)
    #
    # for config in noisy_label_configs:
    #     noisy_mnist = NoisyMNIST(root='/media/amidemo/Data/object_classifier_data/datasets',
    #                                 train=True,
    #                                 labels_noise_perc=config,
    #                                 use_cached_dataset=False)

    noisy_mnist = NoisyMNIST(root='/media/amidemo/Data/object_classifier_data/datasets',
                                   train=True, use_cached_dataset=False,
                                   labels_noise_perc={'symmetric': 0.2})
    pass


def save_noisy_mnist_subset_configs():
    noisy_mnist = NoisyMNISTSubset(root='/media/amidemo/Data/object_classifier_data/datasets',
                                   train=True, use_cached_dataset=True,
                                   labels_noise_perc={8: 1.0}, included_targets={6, 8, 9}
                                   )

    noisy_mnist = NoisyMNISTSubset(root='/media/amidemo/Data/object_classifier_data/datasets',
                                   train=True, use_cached_dataset=True,
                                   labels_noise_perc={5: 1.0}, included_targets={0, 1, 5}
                                   )
    pass


if __name__ == '__main__':
    save_noisy_mnist_configs()
    # save_noisy_mnist_subset_configs()
