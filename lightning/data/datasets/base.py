import os
import pickle
from abc import ABC, abstractmethod
import copy
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Dict

import pandas as pd
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from lightning.data.dataset_utils import pack_decoupled_data, disturb_records_targets


class DatasetExt(ABC):
    """
    The purpose of this class is to serve as base class that provides an alternative to the default __get_item__ method
    which exists inside datasets that inherit from VisionDataset. This is achieved by proper order in multiple
    inheritance
    """

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.get_pil_image()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        sample = {'image': img, 'target': target, 'data_idx': index}
        return sample

    def get_pil_image(self, img):
        return TF.to_pil_image(img)


class RecordsDataset(ABC):
    def __getitem__(self, index: int) -> Any:
        data_entry = copy.copy(self.data[index])

        image = self.get_pil_image(data_entry['image'])

        if self.transform is not None:
            image = self.transform(image)

        data_entry['image'] = image
        data_entry['data_idx'] = index

        return data_entry

    def get_pil_image(self, img):
        return TF.to_pil_image(img)

    def convert_to_records(self):
        # Transform data into list of records (dictionaries)
        self.data = pack_decoupled_data(image=self.data, target=self.targets)


class RecordsDatasetWithNoise(RecordsDataset):
    def __init__(self, **kwargs):
        self._noisy_targets_filepath = None

    def save_noisy_data(self):
        data_df = pd.DataFrame(self.data)
        targets = data_df['target']
        noisy_indices = data_df['is_noisy']
        noisy_data = \
            {
                'targets': targets,
                'noisy_indices': noisy_indices
            }

        self._noisy_targets_filepath.parent.mkdir(exist_ok=True)
        with open(self._noisy_targets_filepath, 'wb') as f:
            pickle.dump(noisy_data, f)

    def set_test_noisy_data(self):
        """
        In test phase none of the data are noisy, therefore we simply set 'is_noisy' to zeros

        @return: None
        """
        data_df = pd.DataFrame(self.data)

        # None of the data are noisy
        noisy_indices = torch.zeros((len(self.data),), dtype=torch.bool)
        data_df['is_noisy'] = noisy_indices.numpy()

        self.data = data_df.to_dict('records')

    def load_noisy_data(self):
        """
        Loads existing noisy targets to the record

        @return:
        """
        with open(self._noisy_targets_filepath, 'rb') as f:
            noisy_data = pickle.load(f)

        data_df = pd.DataFrame(self.data)
        data_df['target'] = noisy_data['targets']
        data_df['is_noisy'] = noisy_data['noisy_indices']
        self.data = data_df.to_dict('records')

    @property
    def dataset_extras_folder(self) -> str:
        """
        In contrast to the above, we keep extra files, e.g. the noisy targets, in a folder named after the name of the
        class.
        @return:
        """
        return os.path.join(self.root, self.__class__.__name__)

    def initialize_noisy_targets_filepath(self):
        # Utilize a predefined filepath which is formed according to the noise percentage
        self._noisy_targets_filepath = \
            Path(self.dataset_extras_folder) / f'{self.__class__.__name__}_train_targets_opt-{self.filename_tag}.pkl'

    def initialize_noisy_data(self, train=False):
        if train:
            if self._use_cached_dataset:
                print(f'Loading from {self._noisy_targets_filepath}')
                self.load_noisy_data()
            else:
                print(f'Creating {self.__class__.__name__} noisy targets {self._labels_noise_perc}')
                self.data = disturb_records_targets(self.data, self._labels_noise_perc, key_to_disturb='target')

                print(f'Saving targets and indices {self._noisy_targets_filepath}')
                self.save_noisy_data()
        else:
            self.set_test_noisy_data()

    @property
    def filename_tag(self):
        raise NotImplementedError
