import copy
import warnings
from typing import Any, Optional, Callable

import torch
from torchvision.datasets import VisionDataset
from pathlib import Path
import csv
from lightning.data.dataset_utils import pil_loader
import pandas as pd
from utils import count_populations
from lightning.ext import logger


class Cars(VisionDataset):
    """
    Standford Cars dataset containing 196 classes. Same dataset appears in torchvision.datasets as StanfordCars, but
    data are formed for classification and not for metric learning. Furthermore this class supports loading of
    noisy variants
    """
    def __init__(self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            ignore_missing_files=True,
            training_variant=None,
            ignore_noisy_samples: bool = False):

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train
        self.training_variant = training_variant
        self.ignore_missing_files = ignore_missing_files
        self.ignore_noisy_samples = ignore_noisy_samples

        self.data = self._load_data()

    def _get_data_file(self, with_noise=True):
        if not self.train:
            return self.raw_folder / 'CARS_test.csv'
        elif self.training_variant is None or with_noise is False:
            return self.raw_folder / 'CARS_train.csv'
        else:
            data_file = self.raw_folder / '{}_train.csv'.format(self.training_variant)
            if not data_file.exists():
                raise AssertionError(f'Invalid training variant {self.training_variant} or missing file {data_file}.')
            return data_file

    def _load_data(self):
        data = []

        self.classes = {}

        data_file = self._get_data_file()
        data_file_orig = self._get_data_file(with_noise=False)

        logger.info(f'Loading data from {data_file}. Also from {data_file_orig} to find noisy indices')

        small_cluster_noise_variant = True if 'smallclusternoised' in data_file.name else False

        # class_name to noisy info
        if small_cluster_noise_variant:
            cleaned_data_file = Path(str(data_file).replace('smallclusternoised', 'smallclusternoised_cleaned'))
            with open(cleaned_data_file, 'r') as f:
                lines = f.readlines()
                # get class name
                clean_classes = set(map(lambda line: line.split(',')[0].split('/')[-2], lines))

        with open(data_file, 'r') as f, open(data_file_orig, 'r') as f_orig:
            csvreader = csv.reader(f)
            csvreader_orig = csv.reader(f_orig)

            for i, (row, row_orig) in enumerate(zip(csvreader, csvreader_orig)):
                image_path, class_id = row
                image_path_orig, class_id_orig = row_orig

                img_path = Path(image_path)
                img_path_orig = Path(image_path)

                if not img_path.exists():
                    if self.ignore_missing_files:
                        continue
                    else:
                        raise FileNotFoundError(f'File {img_path} is missing')

                target = int(class_id)
                target_orig = int(class_id_orig)

                # relabel to start from zero
                if not small_cluster_noise_variant:
                    target -= 1
                    target_orig -= 1

                data_entry = \
                    {
                        'image_path': str(image_path),
                        'target': target,
                        'target_orig': target_orig,
                    }

                class_name = img_path_orig.parent.name

                if not small_cluster_noise_variant:
                    data_entry['is_noisy'] = target != target_orig
                else:
                    data_entry['is_noisy'] = class_name not in clean_classes

                if self.ignore_noisy_samples and data_entry['is_noisy']:
                    continue

                data.append(data_entry)

                if target_orig not in self.classes:
                    self.classes[target_orig] = class_name

            # TODO: incorrect for small cluster variants / not used currently (?)
            self.num_classes = len(self.classes)

        return data

    def __getitem__(self, index: int) -> Any:
        # Get a copy of the entry, such that the original data are not affected below / need to keep only shallow info
        # and not the actual image data. For now, we do shallow copy because we expect that none of the data are nested.
        data_entry = copy.copy(self.data[index])

        image_path = data_entry['image_path']

        image = pil_loader(image_path)

        if self.transform is not None:
            # TODO: decide how to pass the target (label) for possible "relabel transform"
            image = self.transform(image)

        data_entry['image'] = image
        data_entry['data_idx'] = index

        return data_entry

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> Path:
        return Path(self.root) / f'cars196'


def pass_dataset():
    variants = [
            'CARS_0.1noised_cleaned',
            'CARS_0.1noised',
            'CARS_0.25smallclusternoised_cleaned',
            'CARS_0.25smallclusternoised',
            'CARS_0.2noised_cleaned',
            'CARS_0.2noised',
            'CARS_0.5noised_cleaned',
            'CARS_0.5noised',
            'CARS_0.5smallclusternoised_cleaned',
            'CARS_0.5smallclusternoised',
        ]

    for variant in variants:
        dataset = Cars('/data/datasets/', train=True,
                       ignore_missing_files=True, training_variant=variant)


        data_df = pd.DataFrame(dataset.data)
        targets = data_df['target'].values
        populations, ids = count_populations(torch.tensor(targets))
        # print(populations)
        print(variant, len(torch.unique(torch.tensor(targets))))
        pass


def inspect_small_cluster_noise():
    filepath = '/media/amidemo/Data/object_classifier_data/datasets/cars196/CARS_0.5smallclusternoised_train.csv'
    df = pd.read_csv(filepath)
    pass


def inspect_noisy_dataset():
    variant = 'CARS_0.5smallclusternoised'
    noisy_dataset = Cars('/data/datasets/', train=True, ignore_missing_files=True, training_variant=variant)
    clean_dataset = Cars('/data/datasets/', train=True, ignore_missing_files=True, training_variant=None)

    for entry in noisy_dataset.data:
        if noisy_dataset.classes[entry['target']] not in entry['image_path']:
            assert entry['is_noisy'] is True
        else:
            assert entry['is_noisy'] is False

    print(clean_dataset.classes)


def print_class_names():
    train_dataset = Cars('/data/datasets/', train=True, ignore_missing_files=True)
    print('Training classes')
    print('\n'.join(map(lambda kv: f'{kv[0]}: {kv[1]}', train_dataset.classes.items())))
    test_dataset = Cars('/data/datasets/', train=False, ignore_missing_files=True, training_variant=None)
    print('Test classes')
    print('\n'.join(map(lambda kv: f'{kv[0] - 98}: {kv[1]}', test_dataset.classes.items())))


if __name__ == '__main__':
    # inspect_small_cluster_noise()
    # pass_dataset()
    inspect_noisy_dataset()
    # print_class_names()
