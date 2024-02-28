from typing import Any, Callable, Union, Optional, List
import psutil

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pl_bolts.datamodules.vision_datamodule import LightningDataModule

from lightning.data.identity_dataset import Market1501DatasetPart
from lightning.data.samplers import RandomIdentitySampler

from lightning.data.data_modules import get_default_transforms


class IdentityDataModule(LightningDataModule):
    """An alternative VisionDataModule for identity based datasets, e.g. person re-identification"""

    EXTRA_ARGS: dict = {}
    name: str = ""
    #: Dataset class to use
    dataset_cls: type
    #: A tuple describing the shape of the data
    dims: tuple

    def __init__(self,
                 data_dir: Optional[str] = None,
                 num_workers: int = 0,
                 batch_size: int = 32,
                 seed: int = 42,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 training_sampler_class: Optional[type] = None,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:

        if 'sampler_kwargs' in kwargs:
            self.sampler_kwargs = kwargs.pop('sampler_kwargs')
        else:
            self.sampler_kwargs = {}

        if 'transforms' in kwargs:
            # Override deprecated properties with given transforms
            transforms_dict = kwargs.pop('transforms')
        else:
            transforms_dict = {}

        super().__init__(*args, **kwargs)

        # self.dims = (3, 128, 64)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.dataset_train = None
        self.dataset_gallery = None
        self.dataset_query = None

        self._train_transforms = transforms_dict.get('train')
        self._test_transforms = transforms_dict.get('test')

        self.training_sampler_class = training_sampler_class

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """ Normally this contains dummy calls to dataset class in order to download the data. We leave it empty. """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset.

        Currently, val is not implemented, test for identity datasets consists of two parts, gallery and query
        """
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self._train_transforms is None else self._train_transforms

            self.dataset_train = self.dataset_cls(
                self.data_dir, part_name='train', transforms=train_transforms, relabel=True, **self.EXTRA_ARGS)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self._test_transforms is None else self._test_transforms

            self.dataset_gallery = self.dataset_cls(
                self.data_dir, part_name='gallery', transforms=test_transforms, relabel=False, **self.EXTRA_ARGS)
            self.dataset_query = self.dataset_cls(
                self.data_dir, part_name='query', transforms=test_transforms, relabel=False, **self.EXTRA_ARGS)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        if self.training_sampler_class is not None:
            training_sampler = self.training_sampler_class(self.dataset_train, **self.sampler_kwargs)
        else:
            training_sampler = None

        shuffle = self.shuffle and training_sampler is None
        return self._data_loader(self.dataset_train, shuffle=shuffle, sampler=training_sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader. Returns gallery and query data loaders"""
        return [self._data_loader(self.dataset_gallery), self._data_loader(self.dataset_query)]

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError

    def _data_loader(self, dataset: Dataset, shuffle: bool = False, sampler: Sampler = None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=sampler
        )

    def default_transforms(self) -> Callable:
        """
        Omitted check for normalization / prefer to always normalize

        @return: a transforms object
        """
        default_transforms = transforms.Compose([transforms.ToTensor(), imagenet_normalization()])
        return default_transforms


class Market1501DataModule(IdentityDataModule):
    name = "market1501"
    dataset_cls = Market1501DatasetPart
    dims = (3, 128, 64)

    def __init__(self,
                 data_dir: Optional[str] = None,
                 num_workers: int = 0,
                 batch_size: int = 32,
                 seed: int = 42,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 training_sampler_class: Optional[type] = None,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__(data_dir=data_dir, num_workers=num_workers, batch_size=batch_size, seed=seed, shuffle=shuffle,
                         pin_memory=pin_memory, drop_last=drop_last, training_sampler_class=training_sampler_class,
                         *args, **kwargs)
        pass

    @property
    def num_classes(self) -> int:
        """
        Return:
            751
        """
        return 751


if __name__ == '__main__':
    sampler_kwargs = \
        {
            'batch_size': 64,
            'num_instances': 4,
            # the sampler expects only a tuple containing the image and the id, so we create an appropriate function
            'batch_unpack_fn': lambda batch_dict, keys=('image', 'id'): tuple(batch_dict[k] for k in keys)
        }

    dm_args = \
    {
        'data_dir': '/media/amidemo/Data/reid_datasets/market1501/Market-1501-v15.09.15/',
        'batch_size': 64,
        'num_workers': 4,
        'transforms': get_default_transforms('market1501')
    }

    market_dm = Market1501DataModule(seed=13,
                                     shuffle=False,
                                     pin_memory=True,
                                     drop_last=False,
                                     training_sampler_class=RandomIdentitySampler,
                                     sampler_kwargs=sampler_kwargs,
                                     **dm_args
                                     )

    market_dm.setup('fit')
    train_dataloader = market_dm.train_dataloader()
    for batch in train_dataloader:
        #print(batch['id'])
        mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()

        print(f'Memory usage: {100 * mem.used / mem.total:.2f}, '
              f'Swap Memory usage: {100 * swap_mem.used / swap_mem.total:.2f}')
