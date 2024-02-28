from typing import Any, Optional, Union, Dict, Tuple, Set, List
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from dataclasses import dataclass, field, asdict
from lightning.data.datasets import NoisyMNIST, NoisyMNISTSubset


@dataclass
class NoisyMNISTArgs:
    labels_noise_perc: Dict = field(default_factory=lambda: {9: 1.0})
    use_cached_dataset: bool = True


@dataclass
class NoisyMNISTSubsetArgs(NoisyMNISTArgs):
    included_targets: List[int] = field(default_factory=lambda: [0, 1, 2])


class NoisyMNISTDataModule(MNISTDataModule):
    name: str = "noisymnist"
    dataset_cls = NoisyMNIST

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
            dataset_args: NoisyMNISTArgs = None,
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
            self.EXTRA_ARGS = asdict(dataset_args)

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        # super method calls the dataset to ensure data are downloaded. We don't need it here
        pass


class NoisyMNISTSubsetDataModule(NoisyMNISTDataModule):
    name: str = "noisymnistsubset"
    dataset_cls = NoisyMNISTSubset

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
            dataset_args: NoisyMNISTSubsetArgs = None,
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
            dataset_args,
            *args,
            **kwargs,
        )

    def num_classes(self) -> int:
        return len(self.EXTRA_ARGS['included_targets'])
