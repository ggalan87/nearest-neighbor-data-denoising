import os
from pathlib import Path
import random
from typing import Optional, Callable, Tuple, Dict, Literal
import itertools

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.datasets import MNIST, EMNIST
import torchvision.transforms.functional as TF

from lightning.data.datasets.base import DatasetExt
from lightning.data.dataset_utils import generate_random_keep
from lightning.data.transforms import RandomMinMaxTranslate


class DirtyMNIST(MNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 dirty_probability: float = 0.1,
                 translation_limits: Tuple[float, float] = (0.1, 0.5),
                 use_cached_dataset: bool = True,
                 dirtiness_source: Literal['self', 'emnist-l'] = 'self'
                 ):
        super().__init__(root, train, transform, target_transform, download)
        self._dirty_probability = dirty_probability
        self._translation_limits = translation_limits
        self._use_cached_dataset = use_cached_dataset
        self._dirty_samples = torch.zeros_like(self.targets)

        self._dirtiness_source = dirtiness_source

        if self._dirtiness_source == 'self':
            dirtiness_dataset = self
        elif self._dirtiness_source == 'emnist-l':
            dirtiness_dataset = EMNIST(root, split='letters', download=True)
        else:
            raise AssertionError(f'Invalid option for dirtiness_source {self._dirtiness_source}')

        self._dirtiness_data = dirtiness_dataset.data
        self._dirtiness_targets = dirtiness_dataset.targets

        part = 'trainval' if train else 'test'
        # Append dirtiness source in the filename
        d_source = f'_{self._dirtiness_source}' if self._dirtiness_source != 'self' else ''
        lmin, lmax = self._translation_limits

        root_ext = Path(self.dataset_extras_folder)
        self._cached_filepath = \
            root_ext / f'dirty_mnist{d_source}_{part}_dp-{dirty_probability}_limits{lmin}-{lmax}-data.pt'
        self._cached_filepath_indices = \
            root_ext / f'dirty_mnist{d_source}_{part}_dp-{dirty_probability}_limits{lmin}-{lmax}-indices.pt'

        if self._use_cached_dataset:
            # Load the dataset
            print(f'Loading from {self._cached_filepath}')
            self.data = torch.load(self._cached_filepath)
            self._dirty_samples = torch.load(self._cached_filepath_indices)

    def __getitem__(self, index: int) -> Dict:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if not self._use_cached_dataset:
            if random.random() < self._dirty_probability:
                img, target = self.get_dirty_image(index), \
                              self.targets[index]
                self._dirty_samples[index] = True
            else:
                img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        target = int(target)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = TF.to_pil_image(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        sample = {'image': img, 'target': target, 'data_idx': index}
        return sample

    def get_dirty_image(self, dataset_idx: int) -> torch.Tensor:

        # Get the raw data and not from __get_item__()
        image, label = self.data[dataset_idx], int(self.targets[dataset_idx])

        if len(image.shape) != 2:
            raise AssertionError('Only grayscale images are supported for now')

        if self._dirtiness_source == 'self':
            indices_of_rest_labels = torch.where(self._dirtiness_targets != label)[0]
        else:
            # All indices are possible for selection
            indices_of_rest_labels = torch.arange(len(self._dirtiness_targets))

        random_idx = random.choice(indices_of_rest_labels)

        image_random, label_random = self._dirtiness_data[random_idx], self._dirtiness_targets[random_idx]

        image_random = RandomMinMaxTranslate(self._translation_limits)(torch.unsqueeze(image_random, 0))
        return torch.bitwise_xor(image, torch.squeeze(image_random))

    @property
    def dirty_samples(self) -> torch.Tensor:
        return self._dirty_samples

    @property
    def raw_folder(self) -> str:
        """
        Override the default implementation which takes class name as output path, because the dataset is exactly
        the same as the original.
        """
        return os.path.join(self.root, self.__class__.__name__.replace('Dirty', ''), 'raw')

    @property
    def dataset_extras_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def cached_filepath(self):
        return self._cached_filepath

    @property
    def cached_filepath_indices(self):
        return self._cached_filepath_indices

# The Ext version is the same because all required functionality is already implemented in the original dataset, however
# I provide an Ext alias for completeness with other datasets
DirtyMNISTExt = DirtyMNIST


def obtain_dirty_label_configs(n_classes: int, n_excluded: Tuple, noisiness_perc: float = 1.0):
    configs = []
    for n in n_excluded:
        noisy_labels_combos = list(itertools.combinations(range(n_classes), n))
        configs.extend([{l: noisiness_perc for l in combo} for combo in noisy_labels_combos])
    return configs


if __name__ == '__main__':
    def store_dirtymnist_cached():
        dataset = DirtyMNIST(root='/media/amidemo/Data/object_classifier_data/datasets',
                             train=False, dirty_probability=0.1,
                             use_cached_dataset=False, dirtiness_source='emnist-l')

        dirty_images = []

        for i in range(len(dataset)):
            if random.random() < dataset._dirty_probability:
                img = dataset.get_dirty_image(i)
                dataset.dirty_samples[i] = True
            else:
                img = dataset.data[i]
            dirty_images.append(img)
            if i % 1000 == 0:
                print(i)

        torch.save(torch.stack(dirty_images), dataset._cached_filepath)
        torch.save(dataset.dirty_samples, dataset._cached_filepath_indices)

    def generate_random_dirty_images():
        dataset = DirtyMNIST(root='/media/amidemo/Data/object_classifier_data/datasets',
                             train=True, dirty_probability=0.1,
                             use_cached_dataset=True, dirtiness_source='emnist-l')

        dirty_indices = torch.where(dataset.dirty_samples)[0]
        n_dirty = len(dirty_indices)
        shuffled_dirty = torch.randperm(n_dirty)

        border_size = 1
        size = 28 + border_size*2
        tiled_imgs = np.ones((10*size, 10*size, 3))

        for i in range(100):
            idx = dirty_indices[shuffled_dirty[i]]
            im = dataset.data[idx].numpy()
            #im.save(f'/media/amidemo/Data/object_classifier_data/images/mnist/dirty_sample_{i}.jpg')
            r = int(i / 10)
            c = int(i % 10)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            constant = cv2.copyMakeBorder(im, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255, 0, 255])
            tiled_imgs[r*size:r*size+size, c*size:c*size+size, :] = constant

        cv2.imwrite(f'/media/amidemo/Data/object_classifier_data/images/mnist/dirty_sample_tiled_letters.jpg', tiled_imgs)


    def pass_dataset():
        # dataset = DirtyMNIST(root='/media/amidemo/Data/object_classifier_data/datasets',
        #                      train=True, dirty_probability=0.1,
        #                      use_cached_dataset=True, dirtiness_source='emnist-l')
        dataset = MNIST(root='/media/amidemo/Data/object_classifier_data/datasets',
                             train=True)

        elem = dataset[0]
        pass

    pass_dataset()

    # store_dirtymnist_cached()
    # generate_random_dirty_images()
    # test_dirty_mnist_save()
    # test_noisy_mnist()
