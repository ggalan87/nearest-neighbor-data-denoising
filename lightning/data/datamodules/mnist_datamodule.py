from typing import List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import pl_bolts.datamodules.mnist_datamodule as mnist_datamodule
from lightning.data.datasets import MNIST, MNISTSubset


class MNISTDataModule(mnist_datamodule.MNISTDataModule):
    dataset_cls = MNIST

@dataclass
class MNISTSubsetArgs:
    included_targets: List[int] = field(default_factory=lambda: [0, 1, 2])


class MNISTSubsetDataModule(MNISTDataModule):
    name: str = "mnistsubset"
    dataset_cls = MNISTSubset

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
            dataset_args: MNISTSubsetArgs = None,
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

    def num_classes(self) -> int:
        return len(self.EXTRA_ARGS['included_targets'])


def pass_dataloader():
    pass


if __name__ == '__main__':
    pass_dataloader()
