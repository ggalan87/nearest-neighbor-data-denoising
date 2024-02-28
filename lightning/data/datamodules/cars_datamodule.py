from typing import Any, Optional, Union
from dataclasses import dataclass, field, asdict
from torchvision import transforms
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from lightning.data.data_modules import patch_visiondatamodule
from lightning.data.samplers import RandomIdentitySampler
from lightning.data.default_transforms import CarsTrainTransforms, CarsTestTransforms
from lightning.data.datasets import Cars


@dataclass
class CarsArgs:
    training_variant: Optional[str] = None
    ignore_noisy_samples: bool = False


class CarsDataModule(VisionDataModule):
    name: str = "cars"
    dataset_cls = Cars

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
            dataset_args: CarsArgs = None,
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
            if '0.5smallclusternoised' in dataset_args.training_variant:
                self.num_classes = 48
            elif '0.25smallclusternoised' in dataset_args.training_variant:
                self.num_classes = 73
            else:
                self.num_classes = 98
        else:
            #
            self.num_classes = 98

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


def pass_dataloader():
    patch_visiondatamodule(
        sampler_class=RandomIdentitySampler,
        batch_size=64,
        num_instances=8,
        id_key='target',
        fix_samples=True,
        num_epochs=35,
    )

    dm = CarsDataModule(data_dir="/media/amidemo/Data/object_classifier_data/datasets",
                        batch_size=64,
                        train_transforms=CarsTrainTransforms(),
                        val_transforms=CarsTestTransforms(),
                        test_transforms=CarsTestTransforms(),
                        val_split=0.0,
                        dataset_args=CarsArgs(training_variant='CARS_0.5noised'))
    dm.setup('fit')

    train_dataloader = dm.train_dataloader()

    sampler = train_dataloader.sampler

    print(sampler.n_iterations)


if __name__ == '__main__':
    pass_dataloader()
