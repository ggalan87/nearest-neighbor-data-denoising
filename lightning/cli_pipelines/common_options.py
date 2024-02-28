import os
import warnings
import inspect
from typing import Dict
from lightning_lite.utilities.seed import seed_everything
import pl_bolts.datamodules
# local imports
from lightning.models import simple_model, model_base, pcb, resnet_mnist, resnet, inception, unicom
from lightning.data.data_modules import patch_visiondatamodule
import lightning.data.datamodules
from lightning.data.samplers import RandomIdentitySampler
import lightning.data.datasets
from lightning.cli_pipelines.default_settings import *
from lightning.data.default_transforms import *

default_transforms = {
    'MNIST':
        {
            'train': MNISTTrainTransforms(),
            'test': MNISTTestTransforms()
        },
    'KMNIST':
        {
            'train': MNISTTrainTransforms(),
            'test': MNISTTestTransforms()
        },
    'FashionMNIST':
        {
            'train': FashionMNISTTrainTransforms(),
            'test': FashionMNISTTestTransforms()
        },
    'Cars':
        {
            'train': CarsTrainTransforms(),
            'test': CarsTestTransforms()
        },
    'Cars98N':
        {
            'train': Cars98NTrainTransforms(),
            'test': Cars98NTestTransforms()
        },
    'Birds':
        {
            'train': BirdsTrainTransforms(),
            'test': BirdsTestTransforms()
        },
    'OnlineProducts':
        {
            'train': OnlineProductsTrainTransforms(),
            'test': OnlineProductsTestTransforms()
        },
    'Food101N':
        {
            'train': Food101NTrainTransforms(),
            'test': Food101NTestTransforms()
        },
    'CIFAR':
        {
            'train': CIFARTrainTransforms(),
            'test': CIFARTestTransforms()
        }
}

default_transforms['DirtyMNIST'] = default_transforms['MNIST']
default_transforms['NoisyMNIST'] = default_transforms['MNIST']
default_transforms['NoisyMNISTSubset'] = default_transforms['MNIST']
default_transforms['CIFAR10'] = default_transforms['CIFAR']
default_transforms['NoisyCIFAR10'] = default_transforms['CIFAR']


def find_dm_class_definition(dataset_name):
    dm_name = f'{dataset_name}DataModule'
    available_locations = \
        [
            lightning.data.datamodules,
            pl_bolts.datamodules,
        ]

    for module in available_locations:
        class_ = getattr(module, dm_name, None)
        if class_ is not None:
            return class_
    else:
        raise NotImplementedError(f'{dm_name} was not found in {available_locations}')


def find_da_class_definition(dataset_name):
    dm_name = f'{dataset_name}Args'
    available_locations = \
        [
            lightning.data.datamodules,
        ]

    for module in available_locations:
        class_ = getattr(module, dm_name, None)
        if class_ is not None:
            return class_
    else:
        raise NotImplementedError(f'{dm_name} was not found in {available_locations}')


def find_model_class_definition(model_class_name):
    available_locations = \
        [
            simple_model,
            model_base,
            pcb,
            resnet_mnist,
            resnet,
            inception,
            unicom,
        ]

    for module in available_locations:
        class_ = getattr(module, model_class_name, None)
        if class_ is not None:
            return class_
    else:
        raise NotImplementedError(f'{model_class_name} was not found in {available_locations}')


def bootstrap_datamodule(dataset_name: str, sampler_args: Optional[Dict], dataset_args: Dict = None,
                         batch_size: int = None):
    seed_everything(13)

    # Init DataModule
    transforms = default_transforms.get(dataset_name)
    if transforms is None:
        transforms = {}
        warnings.warn('Started with None transforms')

    # Dynamically construct the class from the class name and the corresponding implementations in pl_bolts and locally
    class_ = find_dm_class_definition(dataset_name)

    # Override with <dataset_name>Ext variant whose __get_item__ method returns batch in dict instead of tuple
    # class_.dataset_cls = getattr(lightning.data.datasets, f'{dataset_name}Ext')

    # We need to patch VisionDataModule to be able to set custom Sampler. Sampler args are passed as keyword arguments
    if sampler_args is not None:
        patch_visiondatamodule(sampler_class=RandomIdentitySampler, **sampler_args)

    batch_size = batch_size if batch_size is not None else BATCH_SIZE
    if 'dataset_args' in inspect.signature(class_.__init__).parameters:
        args_class_ = find_da_class_definition(dataset_name)

        # Initialize data module
        args = args_class_(**dataset_args)
        dm = class_(PATH_DATASETS, num_workers=os.cpu_count() - 1, batch_size=batch_size,
                    train_transforms=transforms.get('train'),
                    val_transforms=transforms.get('test'),
                    test_transforms=transforms.get('test'), dataset_args=args)
    else:
        dm = class_(PATH_DATASETS, num_workers=os.cpu_count() - 1, batch_size=batch_size,
                    train_transforms=transforms.get('train'),
                    val_transforms=transforms.get('test'),
                    test_transforms=transforms.get('test'))

    return dm

