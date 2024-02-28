from typing import Any, Optional, Union, Dict, Tuple
from types import MappingProxyType
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization
from dataclasses import dataclass, field, asdict
import lightning.data.samplers
from lightning.data.datasets import NoisyMNIST, DirtyMNISTExt, Cars98N
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

"""Here I define some extra data modules that were not available in pl_bolts. I also define some transforms."""


def get_default_transforms(dataset_class: str):
    """
    Gets universal default transforms for simple datasets. TODO: Need to refactor for supporting per part transforms

    @param dataset_class: the name of the dataset / expressed as in dataset class name
    @return: a single transforms object
    """
    default_transforms = {
        'CIFAR10':
            {
                'train':
                    transforms.Compose(
                        [
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            cifar10_normalization(),
                        ]
                    ),
                'test':
                    transforms.Compose(
                        [
                            transforms.ToTensor(),
                            cifar10_normalization(),
                        ]
                    )
            },
        'market1501':
            {
                'train':
                    transforms.Compose(
                        [
                            transforms.Resize((384, 128)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            imagenet_normalization(),
                        ]
                    ),
                'test':
                    transforms.Compose(
                        [
                            transforms.Resize((384, 128)),
                            transforms.ToTensor(),
                            imagenet_normalization(),
                        ]
                    )
            },
        'messytable':
            {
                'train':
                    transforms.Compose(
                        [
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                        ]
                    ),
                'test':
                    transforms.Compose(
                        [
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                        ]
                    )
            },

    }

    return default_transforms[dataset_class]


def patch_visiondatamodule(**kwargs):
    """
    Patches VisionDataModule in order to expose unexposed functionality such as the sampler of the DataLoader
    This approach is a bit clunky, but otherwise we had to subclass each existing DataModule separately

    @param kwargs: Arbitrary keyword arguments
    @return: -
    """
    sampler_class = kwargs.pop('sampler_class')

    # TODO: Consider other args, e.g. num_workers in the patch
    def train_dataloader(self):
        sampler = sampler_class(self.dataset_train, **kwargs) if sampler_class is not None else None

        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    setattr(VisionDataModule, 'train_dataloader', train_dataloader)
    setattr(VisionDataModule, 'val_dataloader', val_dataloader)
    setattr(VisionDataModule, 'test_dataloader', test_dataloader)



