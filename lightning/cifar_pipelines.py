import os
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import seed_everything
from torch.utils.data.sampler import SequentialSampler
from models.models import LitResnet
from lightning.data.samplers import RandomIdentitySampler
from lightning.data.data_modules import get_default_transforms, patch_visiondatamodule
from lightning.data.datasets import CIFAR10Ext

seed_everything(13)

NUM_WORKERS = int(os.cpu_count() / 2)

from pipelines.pipelines import *
from lightning.cli_pipelines.default_settings import *


def run_training_pipeline(pipeline_options: Dict):
    # 1. Initialize datamodule
    patch_visiondatamodule(sampler_class=RandomIdentitySampler, batch_size=BATCH_SIZE, num_instances=32,
                           batch_unpack_fn=pipeline_options['batch_unpack_fn'])
    cifar10_dm = CIFAR10DataModule(**pipeline_options['dataset_module_args'])

    model = LitResnet(lr=0.05, batch_size=BATCH_SIZE, num_classes=cifar10_dm.num_classes,
                      loss_args={'loss_weight': 1.0, 'loss_warm_up_epochs': 10,
                                 'semi_hard_warm_up_epochs': 30, 'population_warm_up_epochs': 40})

    training_pipeline(model, cifar10_dm, evaluate=True, max_epochs=60, gpus=AVAIL_GPUS)


def run_testing_pipeline(pipeline_options: Dict):
    # 1. Initialize datamodule / for testing we need a sequential sampler for all
    patch_visiondatamodule(sampler_class=SequentialSampler)
    cifar10_dm = CIFAR10DataModule(**pipeline_options['dataset_module_args'])

    weight_path = pipeline_options['weight_path']
    model = LitResnet.load_from_checkpoint(weight_path, batch_unpack_fn=pipeline_options['batch_unpack_fn'])

    testing_pipeline(cifar10_dm, model, extract_features=True, visualize=True, evaluate=False, eval_type='accuracy',
                     gpus=AVAIL_GPUS, **pipeline_options)


def run_model_visualization_pipeline(pipeline_options: Dict):
    patch_visiondatamodule(sampler_class=SequentialSampler)
    cifar10_dm = CIFAR10DataModule(**pipeline_options['dataset_module_args'])

    weight_path = pipeline_options['weight_path']
    model = LitResnet.load_from_checkpoint(weight_path, batch_unpack_fn=pipeline_options['batch_unpack_fn'])

    model_visualization_pipeline(model, cifar10_dm, verbose=False)

    # checkpoints_template = f'./lightning_logs/{cifar10_dm.name}_LitResnet/default/version_{{}}/checkpoints/epoch=59-step=9051.ckpt'
    # checkpoints = [checkpoints_template.format(ver) for ver in range(2, 5)]
    # aligned_visualization_pipeline(data_module=cifar10_dm, checkpoints_paths=checkpoints, model_class=LitResnet)


def run_dataset_visualization_pipeline(pipeline_options: Dict):
    cifar10_dm = CIFAR10DataModule(**pipeline_options['dataset_module_args'])
    dataset_visualization_pipeline(cifar10_dm)

# which dataset
dataset_name = 'CIFAR10'

transforms = get_default_transforms(dataset_name)

# Option to use the Ext variant which returns batch in dict instead of tuple
CIFAR10DataModule.dataset_cls = CIFAR10Ext

if 'Ext' in CIFAR10DataModule.dataset_cls.__name__:
    batch_unpack_fn = lambda batch_dict, keys=('image', 'target', 'data_idx'): tuple(batch_dict[k] for k in keys)
else:
    batch_unpack_fn = None

common_options = \
    {
        'dataset_module_args':
            {
                'data_dir': PATH_DATASETS,
                'batch_size': BATCH_SIZE,
                'num_workers': NUM_WORKERS,
                'train_transforms': transforms['train'],
                'test_transforms': transforms['test'],
                'val_transforms': transforms['test'],
            },
        'batch_unpack_fn': batch_unpack_fn
    }

testing_options = \
    {
        'weight_path': './lightning_logs/cifar10_LitResnet/default/version_2/checkpoints/epoch=59-step=9051.ckpt',
        'output_path': '/media/amidemo/Data/object_image_viz',
        'images_server': 'http://amihome2-ubuntu:8345',
        'dataset_parts': ('test', )
    }

run_testing_pipeline(pipeline_options={**common_options, **testing_options})

# run_dataset_visualization_pipeline(pipeline_options=common_options)
# https://torch.classcat.com/2021/02/25/pytorch-lightning-1-1-research-cifar100-googlenet/
# https://github.com/mikwieczorek/centroids-reid
