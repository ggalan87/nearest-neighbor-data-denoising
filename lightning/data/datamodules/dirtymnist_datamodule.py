from typing import Any, Optional, Union, Dict, Tuple
from dataclasses import dataclass, field, asdict
from torchvision import transforms
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from lightning.data.datasets import DirtyMNIST

@dataclass
class DirtyMNISTArgs:
    dirty_probability: float = 0.1
    translation_limits: Tuple[float, float] = field(default_factory=lambda: (0.1, 0.5))
    use_cached_dataset: bool = True
    dirtiness_source: str = 'self'


class DirtyMNISTDataModule(MNISTDataModule):
    name: str = "dirtymnist"
    dataset_cls = DirtyMNIST

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
            dataset_args: DirtyMNISTArgs = None,
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

