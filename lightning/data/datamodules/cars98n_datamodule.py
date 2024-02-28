from torchvision import transforms
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from lightning.data.datasets import Cars98N


class Cars98NDataModule(VisionDataModule):
    name: str = "cars98n"
    dataset_cls = Cars98N
    num_classes = 98

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
