import copy
from typing import Any, Optional, Callable, Literal
import matplotlib
import cv2
import torch
from PIL import Image
from functools import partial
from numpy.random import default_rng
from torchvision import transforms
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as TF
from pathlib import Path
import shutil
import numpy as np
import pickle
from dataclasses import dataclass
import csv
import yaml
from lightning.data.dataset_utils import disturb_targets, random_split_perc, pack_decoupled_data
from lightning.data.dataset_filter import filter_by
from lightning.data.dataset_generation_utils import *

@dataclass
class ControlledPopulationsDataEntry:
    image: torch.Tensor
    target: int


class ControlledPopulations(VisionDataset):
    classes = [
        "controlled - 0",
        "controlled - 1",
        "controlled - 2",
        "controlled - 2",
        "controlled - 3",
        "controlled - 4",
        "controlled - 5",
        "controlled - 6",
    ]

    def __init__(self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            with_noise=True,
            dataset_part: Literal['train', 'gallery', 'query'] = 'train',
            variant='random_noise'):

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        self.variant = variant

        if train and dataset_part != 'train':
            raise AssertionError(f'Invalid part {dataset_part} while train')

        self.dataset_part = dataset_part

        self.data, self.targets, self.noisy_indices = self._load_data()

        if not with_noise:
            print('Intentionally eliminating noisy data from the dataset!')
            self.targets[self.noisy_indices] = 1
            self.noisy_indices[self.noisy_indices] = False

        if len(self.data[0].shape) == 3:
            self.data = self.data.permute(0, 3, 1, 2)
            self.img_convert_func = partial(TF.to_pil_image, mode=None)
        else:
            self.img_convert_func = partial(TF.to_pil_image, mode='L')

        self.data = pack_decoupled_data(image=self.data, target=self.targets, is_noisy=self.noisy_indices)

        # Late load transform based on config
        self._create_transform()

    def _load_data(self):
        data_path = self.raw_folder / 'trainval.pkl' if self.train else self.raw_folder / f'{self.dataset_part}.pkl'

        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        images, targets, noisy_indices = data_dict['image'], data_dict['target'], data_dict['is_noisy']

        print(f'Loaded data from {data_path}.')
        return images, targets, noisy_indices

    def _create_transform(self):
        dataset_root = Path(self.root) / f'controlled_populations/{self.variant}'
        config_path = dataset_root / 'config.yaml'
        config = yaml.load(open(config_path), yaml.SafeLoader)

        mean = config['dataset_stats']['mean']
        std = config['dataset_stats']['std']

        mean = tuple(mean) if isinstance(mean, list) else (mean, )
        std = tuple(std) if isinstance(std, list) else (std, )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __getitem__(self, index: int) -> Any:
        data_entry = copy.copy(self.data[index])

        image = self.img_convert_func(data_entry['image'])  # TF.to_pil_image(data_entry['image'], mode='L')

        if self.transform is not None:
            image = self.transform(image)

        data_entry['image'] = image
        data_entry['data_idx'] = index

        return data_entry

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self) -> Path:
        return Path(self.root) / f'controlled_populations/{self.variant}'





def pass_dataset():
    dataset = \
        ControlledPopulations('/media/amidemo/Data/object_classifier_data/datasets', train=False, dataset_part='gallery')
    print(len(dataset))


def save_dataset_images():
    gallery_dataset = \
        ControlledPopulations('/media/amidemo/Data/object_classifier_data/datasets', train=False,
                              dataset_part='gallery')

    images_path = Path('/media/amidemo/Data/object_classifier_data/global_plots/images')
    for idx in range(len(gallery_dataset)):
        img = gallery_dataset.data[idx]

        img = TF.to_pil_image(img, mode='L')
        img.save(images_path / f'train_{idx}_{gallery_dataset.targets[idx]}.png')
        #break


def save_grids():
    gallery_dataset = \
        ControlledPopulations('/media/amidemo/Data/object_classifier_data/datasets', train=False,
                              dataset_part='gallery', variant='random_noise_rgb_gradient')

    n_targets = len(torch.unique(gallery_dataset.targets))
    for target in range(n_targets):
        class_targets_indices = torch.where(gallery_dataset.targets == target)[0]
        n_from_class = len(class_targets_indices)
        shuffled_class_indices = torch.randperm(n_from_class)

        border_size = 1
        size = 28 + border_size * 2
        tiled_imgs = np.ones((10 * size, 10 * size, 3))

        for i in range(100):
            idx = class_targets_indices[shuffled_class_indices[i]]
            im = gallery_dataset.data[idx]['image'].numpy()
            r = int(i / 10)
            c = int(i % 10)

            # TODO: Check with channels
            if len(im.shape) == 3:
                # Data are kept in CWH form as pytorch requires so we do a transpose
                im = np.transpose(im, (1, 2, 0))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            else:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


            constant = cv2.copyMakeBorder(im, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                          value=[255, 0, 255])
            tiled_imgs[r * size:r * size + size, c * size:c * size + size, :] = constant

        cv2.imwrite(f'/media/amidemo/Data/object_classifier_data/images/controlled-{gallery_dataset.variant}'
                    f'/controlled-class-{target}.jpg', tiled_imgs)


def convert_to_noisy_indices():
    root = Path('/media/amidemo/Data/object_classifier_data/datasets/controlled_populations/random_noise')
    for p in root.iterdir():
        if p.suffix != '.pkl':
            continue

        data = pickle.load(open(p, 'rb'))

        noisy_indices = data['is_noisy']

        is_noisy = torch.zeros(len(data['target']), dtype=torch.bool)

        if noisy_indices is not None:
            is_noisy[noisy_indices] = True
            assert torch.count_nonzero(is_noisy) == len(noisy_indices)
            print(noisy_indices.min(), noisy_indices.max())

        data['is_noisy'] = is_noisy

        pickle.dump(data, open(p, 'wb'))


def generate_random_noise(overwrite=False):
    root = Path('/media/amidemo/Data/object_classifier_data/datasets/') / 'controlled_populations/random_noise_test'
    generator = GrayscaleNoiseDatasetGenerator(root, overwrite, 'v2')
    generator.generate()


def generate_random_hsv_noise(overwrite=True):
    root = Path('/media/amidemo/Data/object_classifier_data/datasets/') / 'controlled_populations/random_noise_hsv_test'
    generator = HSVNoiseDatasetGenerator(root, overwrite, 'v2')
    generator.generate()


def generate_random_rgb_noise(overwrite=True):
    root = Path('/media/amidemo/Data/object_classifier_data/datasets/') / 'controlled_populations/random_noise_rgb-v2'
    generator = RGBNoiseDatasetGenerator(root, overwrite, 'v2')
    generator.generate()

def generate_random_rgb_gradient_noise(overwrite=True):
    root = Path('/media/amidemo/Data/object_classifier_data/datasets/') / 'controlled_populations/random_noise_rgb_gradient-v1'
    generator = RGBGradientNoiseDatasetGenerator(root, overwrite, 'v1')
    generator.generate()

if __name__ == '__main__':
    # generate_random_noise(overwrite=False)
    # generate_random_hsv_noise(overwrite=True)
    # pass_dataset()
    # generate_random_rgb_noise(overwrite=True)
    generate_random_rgb_gradient_noise(overwrite=True)
    # save_dataset_images()
    save_grids()
    # convert_to_noisy_indices()
