from typing import Any, Optional, Union, Dict, Tuple, Set, List
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from dataclasses import dataclass, field, asdict
from lightning.data.datasets import NoisyCIFAR10


@dataclass
class NoisyCIFARArgs:
    labels_noise_perc: Dict = field(default_factory=lambda: {'symmetric': 0.5})
    use_cached_dataset: bool = True

NoisyCIFAR10Args = NoisyCIFARArgs

class NoisyCIFAR10DataModule(CIFAR10DataModule):
    name: str = "noisycifar10"
    dataset_cls = NoisyCIFAR10

    def __init__(
            self,
            data_dir: Optional[str] = None,
            val_split: Union[int, float] = 0.0,
            num_workers: int = 0,
            normalize: bool = False,
            batch_size: int = 32,
            seed: int = 42,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            dataset_args: NoisyCIFARArgs = None,
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

        if val_split != 0.0:
            raise NotImplementedError('Split of training data into train/val is not implemented yet. '
                                      'It requires resetting noisy labels to original labels, '
                                      'because test data do not have noise')
        # Already inherited / no need to set
        # self.num_classes = 10

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        # super method calls the dataset to ensure data are downloaded. We don't need it here
        pass


def pass_dataset():
    dm = NoisyCIFAR10DataModule(data_dir='/data/datasets', val_split=0.0)
    dm.setup('fit')
    pass


if __name__ == '__main__':
    pass_dataset()