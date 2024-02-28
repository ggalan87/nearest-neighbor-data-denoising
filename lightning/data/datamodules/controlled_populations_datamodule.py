from typing import Any, Optional, Union, Dict, Tuple, List
from torch.utils.data import DataLoader
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from dataclasses import dataclass, field, asdict
from lightning.data.datasets import ControlledPopulations

from lightning.data.dataset_filter import FilterBase


@dataclass
class ControlledPopulationsArgs:
    with_noise: bool = True
    filter: List[FilterBase] = None
    variant: str = 'random_noise'


class ControlledPopulationsDataModule(VisionDataModule):
    name: str = "controlledpopulations"
    dataset_cls = ControlledPopulations

    def __init__(
            self,
            data_dir: Optional[str] = None,
            val_split: Union[int, float] = 0.2,
            num_workers: int = 0,
            normalize: bool = False,
            batch_size: int = 32,
            seed: int = 42,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            dataset_args: ControlledPopulationsArgs = None,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            data_dir,
            val_split,
            num_workers,
            normalize,
            batch_size,
            seed,
            shuffle,
            pin_memory,
            drop_last,
            *args,
            **kwargs,
        )

        if dataset_args:
            args = asdict(dataset_args)
            dataset_filters_list = args.get('filter')

            # Replace the dataset class with same class
            if dataset_filters_list is not None:
                raise NotImplementedError
                # self.dataset_cls
            else:
                args.pop('filter')

            self.EXTRA_ARGS = args

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, gallery, and query dataset."""
        if stage == "fit" or stage is None:
            self.dataset_train = self.dataset_cls(self.data_dir, train=True, transform=None, **self.EXTRA_ARGS)
            self.dataset_val = self.dataset_cls(
                self.data_dir, train=False, transform=None, dataset_part='query', **self.EXTRA_ARGS
            )

        if stage == "test" or stage is None:
            self.dataset_gallery = self.dataset_cls(
                self.data_dir, train=False, transform=None, dataset_part='gallery', **self.EXTRA_ARGS
            )

            self.dataset_query = self.dataset_cls(
                self.data_dir, train=False, transform=None, dataset_part='query', **self.EXTRA_ARGS
            )

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        # Normally contains dummy calls for ensuring downloaded the data, omit for now
        pass

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader. Returns gallery and query data loaders"""
        return [self._data_loader(self.dataset_gallery), self._data_loader(self.dataset_query)]


if __name__ == '__main__':
    dm = ControlledPopulationsDataModule(data_dir='/media/amidemo/Data/object_classifier_data/datasets')
    dm.setup('test')
    pass