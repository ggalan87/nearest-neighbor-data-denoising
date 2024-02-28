import copy
from typing import Any, Optional, Callable

import pandas as pd
import torch
from torchvision.datasets import VisionDataset
from pathlib import Path
import csv
from lightning.data.dataset_utils import pil_loader
from lightning.data.datasets.cars import Cars
from utils import count_populations
from lightning.ext import logger


class Cars98N(Cars):
    """
    A noisy dataset proposed in "Noise-resistant Deep Metric Learning with Ranking-based Instance Selection"
    with 98 car models crawled from pinterest, also containing noisy labels
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):

        super().__init__(root, transform=transform, target_transform=target_transform, download=download)

        self.train = train

        self.data = self._load_data()

    def _load_data(self):
        data = []

        self.classes = {}

        data_file = self._get_data_file()

        with open(data_file, 'r') as f:
            csvreader = csv.reader(f)

            for i, row in enumerate(csvreader):
                image_path, class_id = row

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

                data.append(data_entry)

            self.num_classes = 98

        return data

    def _get_data_file(self, with_noise=True):
        return self.raw_folder / 'CARSN_{}.csv'.format('train' if self.train else 'test')

    @property
    def raw_folder(self) -> Path:
        return Path(self.root) / f'cars_noise'


def pass_dataset():
    dataset = Cars98N('/data/datasets/', train=True)

    data_df = pd.DataFrame(dataset.data)
    targets = data_df['target'].values
    populations, ids = count_populations(torch.tensor(targets))
    pass


if __name__ == '__main__':
    pass_dataset()
