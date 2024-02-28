from typing import Tuple, Optional, Union
from torchvision import transforms
import augly.image as imaugs
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


class MNISTTransforms:
    def __init__(self):
        self.transforms = \
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )

    def __call__(self, img):
        return self.transforms(img)


MNISTTrainTransforms = MNISTTransforms
MNISTTestTransforms = MNISTTransforms


class FashionMNISTTransforms:
    def __init__(self):
        self.transforms = \
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,)),
                ]
            )

    def __call__(self, img):
        return self.transforms(img)


FashionMNISTTrainTransforms = FashionMNISTTransforms
FashionMNISTTestTransforms = FashionMNISTTransforms


# Adapted from https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/cifar10-baseline.html
class CIFARTestTransforms:
    def __init__(self):
        self.transforms = \
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    cifar10_normalization()
                ]
            )

    def __call__(self, img):
        return self.transforms(img)


class CIFARTrainTransforms:
    def __init__(self):
        self.transforms = \
            transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    cifar10_normalization(),
                ]
            )

    def __call__(self, img):
        return self.transforms(img)


class UpscaledCIFARTestTransforms:
    def __init__(self, resize_to: Union[int, Tuple[int, int]] = (256, 256),
                 crop_to: Union[int, Tuple[int, int]] = (224, 224)):
        self.transforms = \
            transforms.Compose(
                [
                    transforms.Resize(size=resize_to),
                    transforms.CenterCrop(size=crop_to),
                    transforms.ToTensor(),
                    cifar10_normalization()
                ]
            )

    def __call__(self, img):
        return self.transforms(img)


class UpscaledCIFARTrainTransforms:
    def __init__(self, resize_to: Union[int, Tuple[int, int]] = (256, 256),
                 crop_to: Union[int, Tuple[int, int]] = (224, 224)):
        self.transforms = \
            transforms.Compose(
                [
                    transforms.Resize(size=resize_to),
                    transforms.CenterCrop(size=crop_to),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    cifar10_normalization(),
                ]
            )

    def __call__(self, img):
        return self.transforms(img)


class BasicTransforms:
    def __init__(self, resize_to: Union[int, Tuple[int, int]] = (256, 256),
                 crop_to: Union[int, Tuple[int, int]] = (224, 224)):
        self.transforms = \
            transforms.Compose(
                [
                    transforms.Resize(size=resize_to),
                    transforms.CenterCrop(size=crop_to),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __call__(self, img):
        return self.transforms(img)


class BasicTrainTransforms(BasicTransforms):
    def __init__(self,
                 resize_to: Union[int, Tuple[int, int]] = (256, 256),
                 crop_to: Union[int, Tuple[int, int]] = (224, 224), 
                 scale_range: Tuple[float, float] = (0.2, 1.0),
                 flip_probability=0.5):
        super().__init__(resize_to, crop_to)
        del self.transforms.transforms[1]
        self.transforms.transforms.insert(1, transforms.RandomResizedCrop(
                        scale=scale_range, size=crop_to
                    ))
        self.transforms.transforms.insert(2, transforms.RandomHorizontalFlip(p=flip_probability))


# Same as above but with clip normalization
# TODO: construct it dynamically, e.g. pass normalization as parameter, or create a function which takes
#  the base class as a parameter
class BasicClipTransforms(BasicTransforms):
    def __init__(self, resize_to: Union[int, Tuple[int, int]] = (256, 256),
                 crop_to: Union[int, Tuple[int, int]] = (224, 224)):
        super().__init__(resize_to=resize_to, crop_to=crop_to)
        self.transforms.transforms[-1] = transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                    )


class BasicClipTrainTransforms(BasicTrainTransforms):
    def __init__(self,
                 resize_to: Union[int, Tuple[int, int]] = (256, 256),
                 crop_to: Union[int, Tuple[int, int]] = (224, 224),
                 scale_range: Tuple[float, float] = (0.2, 1.0),
                 flip_probability=0.5):
        super().__init__(resize_to=resize_to, crop_to=crop_to, scale_range=scale_range,
                         flip_probability=flip_probability)
        self.transforms.transforms[-1] = transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                    )


COLOR_JITTER_PARAMS = {
    "brightness_factor": 1.2,
    "contrast_factor": 1.2,
    "saturation_factor": 1.4,
}

AUGMENTATIONS = [
    imaugs.RandomBrightness(),
    # imaugs.RandomBlur(),
    # imaugs.RandomPixelization(),
    #imaugs.ColorJitter(**COLOR_JITTER_PARAMS),
    # imaugs.OneOf(
    #     [imaugs.OverlayOntoScreenshot(), imaugs.OverlayEmoji(), imaugs.OverlayText()]
    # ),
    #transforms.Lambda(lambda img: img.convert("RGB"))
]


class BasicAugTrainTransforms(BasicTrainTransforms):
    def __init__(self,
                 resize_to: Union[int, Tuple[int, int]] = (256, 256),
                 crop_to: Union[int, Tuple[int, int]] = (224, 224),
                 scale_range: Tuple[float, float] = (0.2, 1.0),
                 flip_probability=0.5):
        super().__init__(resize_to, crop_to, scale_range, flip_probability)

        position = 3
        for aug in AUGMENTATIONS:
            self.transforms.transforms.insert(position, aug)
            position += 1
        pass


CarsTrainTransforms = BasicTrainTransforms
CarsTestTransforms = BasicTransforms

Cars98NTrainTransforms = BasicTrainTransforms
Cars98NTestTransforms = BasicTransforms

BirdsTrainTransforms = BasicTrainTransforms
BirdsAugTrainTransforms = BasicAugTrainTransforms
BirdsTestTransforms = BasicTransforms

OnlineProductsTrainTransforms = BasicTrainTransforms
OnlineProductsTestTransforms = BasicTransforms
OnlineProductsClipTrainTransforms = BasicClipTrainTransforms
OnlineProductsClipTestTransforms = BasicClipTransforms

Food101NTrainTransforms = BasicTrainTransforms
Food101NTestTransforms = BasicTransforms

if __name__ == '__main__':
    train_transforms = Cars98NTrainTransforms()
