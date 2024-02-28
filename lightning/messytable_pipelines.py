import os
from pytorch_lightning import seed_everything
from torch.utils.data.sampler import SequentialSampler
from models.models import LitResnet
from lightning.data.samplers import RandomIdentitySampler
from lightning.data.data_modules import get_default_transforms, patch_visiondatamodule
from lightning.data.custom_data_handling.messytable_dataset import MessyTableDatamodule
from losses import TripletLoss

seed_everything(13)

NUM_WORKERS = int(os.cpu_count() / 2)

from pipelines.pipelines import *
from lightning.cli_pipelines.default_settings import *

from data.dataset_filter import *


def run_training_pipeline(pipeline_options: Dict):
    # 1. Initialize datamodule
    messytable_dm = MessyTableDatamodule(**pipeline_options['dataset_module_args'])


    # TODO: obtain somehow the number of classes in a proper way. needs thinking because I don't have only classes but also groups and subclasses
    model = LitResnet(lr=0.05, batch_size=16, num_classes=10, loss_class=pipeline_options.get('loss_class'),
                      loss_args=pipeline_options.get('loss_args'), batch_unpack_fn=pipeline_options.get('batch_unpack_fn'))

    training_pipeline(model, messytable_dm, evaluate=True, max_epochs=60, gpus=AVAIL_GPUS)


# def run_testing_pipeline(pipeline_options: Dict):
#     # 1. Initialize datamodule / for testing we need a sequential sampler for all
#     patch_visiondatamodule(sampler_class=SequentialSampler)
#     messytable_dm = MessyTableDatamodule(**pipeline_options['dataset_module_args'])
#
#     weight_path = pipeline_options['weight_path']
#     model = LitResnet.load_from_checkpoint(weight_path, batch_unpack_fn=pipeline_options['batch_unpack_fn'])
#
#     testing_pipeline(messytable_dm, model, extract_features=False, visualize=False, evaluate=True, eval_type='reid',
#                      gpus=AVAIL_GPUS, **pipeline_options)

def run_custom_testing_pipeline(pipeline_options: Dict, save_feats=False, eval=False):
    # 1. Initialize datamodule
    market_dm = MessyTableDatamodule(**pipeline_options['dataset_module_args'])

    weight_path = pipeline_options['weight_path']

    if weight_path is not None:
        model = LitResnet.load_from_checkpoint(weight_path, pretrained=False,
                                           batch_unpack_fn=pipeline_options.get('batch_unpack_fn'))
    else:
        # Just load the pretrained model
        model = LitResnet(batch_unpack_fn=pipeline_options.get('batch_unpack_fn'))

    #testing_pipeline(market_dm, model, extract_features=True, visualize=True, evaluate=False, eval_type='accuracy',
    #                 gpus=AVAIL_GPUS, **pipeline_options)

    # Change the model to evaluation mode
    model.eval()

    market_dm.setup('test')

    gallery_dl, query_dl = market_dm.test_dataloader()

    gallery_feats = get_feats(gallery_dl, model)
    query_feats = get_feats(query_dl, model)

    if save_feats:
        storage = features_storage.FeaturesStorage(market_dm.name)

        storage.add('train', gallery_feats)
        storage.add('test', query_feats)

        suffix = pipeline_options.get('save_suffix', '')
        output_filepath = f'./features/{market_dm.name}_{type(model).__name__}{suffix}.pt'
        storage.save(output_filepath)

    if eval:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Computing distance matrix on feats...', gallery_feats['feats'].shape, query_feats['feats'].shape)
        dists_np = compute_distance_matrix(query_feats['feats'].to(device), gallery_feats['feats'].to(device)).detach().cpu().numpy()

        query_labels_np = query_feats['id'].detach().cpu().numpy().ravel()
        gallery_labels_np = gallery_feats['id'].detach().cpu().numpy().ravel()

        query_cids_np = query_feats['view_id'].detach().cpu().numpy().ravel()
        gallery_cids_np = gallery_feats['view_id'].detach().cpu().numpy().ravel()


        print('Running evaluation...')
        all_cmc, mAP = evaluate_rank(dists_np, query_labels_np, gallery_labels_np,
                                     query_cids_np, gallery_cids_np, max_rank=50)
        print(all_cmc, mAP)

    del model


def run_model_visualization_pipeline(pipeline_options: Dict):
    patch_visiondatamodule(sampler_class=SequentialSampler)
    cifar10_dm = MessyTableDatamodule(**pipeline_options['dataset_module_args'])

    weight_path = pipeline_options['weight_path']
    model = LitResnet.load_from_checkpoint(weight_path, batch_unpack_fn=pipeline_options['batch_unpack_fn'])

    model_visualization_pipeline(model, cifar10_dm, verbose=False)

    # checkpoints_template = f'./lightning_logs/{cifar10_dm.name}_LitResnet/default/version_{{}}/checkpoints/epoch=59-step=9051.ckpt'
    # checkpoints = [checkpoints_template.format(ver) for ver in range(2, 5)]
    # aligned_visualization_pipeline(data_module=cifar10_dm, checkpoints_paths=checkpoints, model_class=LitResnet)


def run_dataset_visualization_pipeline(pipeline_options: Dict):
    cifar10_dm = MessyTableDatamodule(**pipeline_options['dataset_module_args'])
    dataset_visualization_pipeline(cifar10_dm)

# which dataset
dataset_name = 'messytable'

transforms = get_default_transforms(dataset_name)

fn_keys = ('image', 'subclass_id')
#fn_keys = ('image', 'id', 'view_id')

batch_unpack_fn = lambda batch_dict, keys=fn_keys: tuple(batch_dict[k] for k in keys)

dataset_filter = \
            [
                LocalRandomKeepFilter(field_name='subclass_id', keep_probability=0.1),
                RangeFilter(field_name='subclass_id', range_=range(0, 10))
            ]

common_options = \
    {
        'dataset_module_args':
            {
                'data_dir': '/media/amidemo/Data/MessyTable',
                'batch_size': 16,
                'num_workers': NUM_WORKERS,
                'transforms': get_default_transforms(dataset_name),
            },
        'batch_unpack_fn': batch_unpack_fn
    }

training_options = \
    {
        'dataset_module_args':
            {
                'training_sampler_class': RandomIdentitySampler,
                'sampler_kwargs':
                    {
                        'batch_size': 16,
                        'num_instances': 4,
                        'batch_unpack_fn': batch_unpack_fn,
                        'id_key': 'subclass_id',
                    },
                # Select subset of dataset
                'dataset_filter': dataset_filter
            },
        'loss_class': TripletLoss,
        'loss_args':
            {
                'loss_weight': 1.0,
            }
    }

testing_options = \
    {
        'weight_path': './lightning_logs/messytable_LitResnet/default/version_0/checkpoints/epoch-59.ckpt',
        'output_path': '/media/amidemo/Data/object_image_viz',
        'images_server': 'http://wall2:8345',
        'dataset_parts': ('test', )
    }

# run_training_pipeline(pipeline_options=merge(common_options, training_options))
# run_testing_pipeline(pipeline_options=merge(common_options, testing_options))

# run_dataset_visualization_pipeline(pipeline_options=common_options)
# https://torch.classcat.com/2021/02/25/pytorch-lightning-1-1-research-cifar100-googlenet/
# https://github.com/mikwieczorek/centroids-reid
