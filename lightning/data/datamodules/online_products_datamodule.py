from typing import Any, Optional, Union
from dataclasses import dataclass, field, asdict
from torchvision import transforms
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from lightning.data.datasets import OnlineProducts


@dataclass
class OnlineProductsArgs:
    training_variant: Optional[str] = None


class OnlineProductsDataModule(VisionDataModule):
    name: str = "onlineproducts"
    dataset_cls = OnlineProducts

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
            dataset_args: OnlineProductsArgs = None,
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

            # Determined with
            # cat <training_variant_filename>.csv | cut -d',' -f2 | sort -n -u | wc -l
            if dataset_args.training_variant is None:
                self.num_classes = 11318
            elif '0.5smallclusternoised' in dataset_args.training_variant:
                self.num_classes = 5674
            elif '0.25smallclusternoised' in dataset_args.training_variant:
                self.num_classes = 8511
            elif '0.5noised' in dataset_args.training_variant:
                self.num_classes = 11249
            elif '0.2noised' in dataset_args.training_variant:
                self.num_classes = 11268
            elif '0.1noised' in dataset_args.training_variant:
                self.num_classes = 11309
            else:
                self.num_classes = 11318
        else:
            #
            self.num_classes = 11318

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        # super method calls the dataset to ensure data are downloaded. We don't need it here
        pass

    def default_transforms(self):
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        transform = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
                normalize_transform,
            ]
        )
        return transform