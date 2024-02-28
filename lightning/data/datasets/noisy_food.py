import copy
import warnings
from typing import Any, Optional, Callable, Literal

import torch
from torchvision.datasets import VisionDataset
from pathlib import Path
import csv
from lightning.data.dataset_utils import pil_loader
import pandas as pd
from tqdm import tqdm
from utils import count_populations
from lightning.ext import logger


class Food101N(VisionDataset):
    """
    Food-101N contains 310,009 images of food recipes in 101 classes. The test set is the Food-101 dataset, which
    contains the same 101 classes as Food-101N
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ignore_missing_files=True,
                 training_variant: Literal['full', 'verified'] = 'full',
                 ignore_noisy_samples: bool = False):

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train
        self.training_variant = training_variant
        self.ignore_missing_files = ignore_missing_files
        self.ignore_noisy_samples = ignore_noisy_samples

        self.data = self._load_data()

    def _get_data_file(self):
        if not self.train:
            return self.raw_folder / 'FOOD_test.csv'
        else:
            if self.training_variant == 'full':
                return self.raw_folder / 'FOOD_train.csv'
            elif self.training_variant == 'verified':
                return self.raw_folder / 'FOOD_train_verified.csv'
            else:
                raise NotImplementedError

    def _load_data(self):
        data = []

        self.classes = {}

        data_file = self._get_data_file()

        print(f'Loading data from {data_file}.')

        with open(data_file, 'r') as f:
            csvreader = csv.reader(f)

            for i, row in enumerate(csvreader):
                if self.training_variant == 'full' or not self.train:
                    image_path, class_id = row
                else:
                    image_path, class_id, verified = row

                img_path = Path(image_path)

                if not img_path.exists():
                    if self.ignore_missing_files:
                        continue
                    else:
                        raise FileNotFoundError(f'File {img_path} is missing')

                target = int(class_id)

                # relabel to start from zero
                target -= 1

                data_entry = \
                    {
                        'image_path': str(image_path),
                        'target': target,
                    }

                if not self.train:
                    data_entry['is_noisy'] = False
                    data_entry['target_orig'] = target
                elif self.training_variant == 'full':
                    warnings.warn('is_noisy information is not available for this dataset, '
                                  'because training images have been crawled from Google. '
                                  'Try the verified variant (subset of the full)')

                else:
                    data_entry['is_noisy'] = verified == '0'

                    # In case of the verified variant we have 50 classes. We additionally consider 51th class as a
                    # general noisy class
                    data_entry['target_orig'] = 50 if data_entry['is_noisy'] else target

                data.append(data_entry)

                class_name = img_path.parent.stem
                if target not in self.classes:
                    self.classes[target] = class_name

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
        return Path(self.root) / f'Food-101N'


def pass_dataset():
    dataset = Food101N('/data/datasets/', train=True, training_variant='verified', ignore_missing_files=True)

    data_df = pd.DataFrame(dataset.data)
    targets = data_df['target'].values

    n_targets = len(torch.unique(torch.tensor(targets)))
    tmax = targets.max()
    print(n_targets, tmax,  (targets.max() + 1) == n_targets)

    populations, ids = count_populations(torch.tensor(targets))
    print(populations)
    pass


def write_verified_train():
    original_train_file_path = Path('/data/food101nDML/FOOD_train.csv')
    verified_train_file_path = Path('/data/food101nDML/Food-101N_release/meta/verified_train.tsv')

    verified_files = []

    with open(verified_train_file_path, 'r') as f:
        tsvreader = csv.reader(f, delimiter='\t')

        next(tsvreader)

        for i, row in enumerate(tsvreader):
            verified_files.append(('/data/food101nDML/Food-101N_release/images/'+row[0], row[1]))

    verified_df = pd.DataFrame(verified_files)
    verified_df.columns = ["img_path", "verified"]

    original_files = []

    with open(original_train_file_path, 'r') as f:
        csvreader = csv.reader(f)

        for i, row in enumerate(csvreader):
            image_path, class_id = row
            original_files.append(row)

    original_df = pd.DataFrame(original_files)
    original_df.columns = ["img_path", "label"]

    result = pd.merge(original_df, verified_df, on="img_path")

    result.to_csv(original_train_file_path.parent / 'FOOD_train_verified.csv', index=False, header=False)


if __name__ == '__main__':
    pass_dataset()
    # write_verified_train()
