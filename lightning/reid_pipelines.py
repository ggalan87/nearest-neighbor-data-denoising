import os
import platform

from pytorch_lightning import seed_everything
from models.models import LitPCB
from lightning.data.samplers import RandomIdentitySampler
from lightning.data.data_modules import get_default_transforms
from lightning.data.datamodules.identity_datamodule import Market1501DataModule
from lightning.losses_playground import PopulationAwareTripletLoss
from features_storage import FeaturesStorage
import inspect
seed_everything(13)

NUM_WORKERS = int(os.cpu_count() / 2)

from pipelines.pipelines import *
from lightning.cli_pipelines.default_settings import *
from utils import merge

def run_training_pipeline(pipeline_options: Dict):
    # 1. Initialize datamodule
    market_dm = Market1501DataModule(**pipeline_options['dataset_module_args'])

    model = LitPCB(lr=0.1, batch_size=market_dm.batch_size, num_classes=market_dm.num_classes,
                   batch_unpack_fn=pipeline_options.get('batch_unpack_fn'),
                   loss_class=pipeline_options.get('loss_class'),
                   loss_args=pipeline_options.get('loss_args'))

    training_pipeline(model, market_dm, evaluate=False, max_epochs=60, gpus=AVAIL_GPUS)


def get_feats(dataloader, model):
    dataset_attributes = \
        {
            'feats': []
        }

    # Get the rest available keys by inspecting the batch_unpack function of the model and more specifically the
    # default
    batch_keys = inspect.getfullargspec(model.batch_unpack_fn).defaults[0]
    for k in batch_keys[1:]:
        dataset_attributes[k] = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def pass_dataloader(dataloader):
        for batch in dataloader:
            images, *rest_info = model.batch_unpack_fn(batch)
            feats = model.forward_features(images.to(device))
            dataset_attributes['feats'].append(feats.detach().cpu())

            for i, k in enumerate(batch_keys[1:]):
                dataset_attributes[k].append(rest_info[i])

    with torch.no_grad():
        pass_dataloader(dataloader)

    for k, v in dataset_attributes.items():
        dataset_attributes[k] = torch.cat(dataset_attributes[k])
    return dataset_attributes


def run_custom_testing_pipeline(pipeline_options: Dict, save_feats=False, eval=False):
    # 1. Initialize datamodule
    market_dm = Market1501DataModule(**pipeline_options['dataset_module_args'])

    weight_path = pipeline_options['weight_path']

    if weight_path is not None:
        model = LitPCB.load_from_checkpoint(weight_path, pretrained=False,
                                           batch_unpack_fn=pipeline_options.get('batch_unpack_fn'))
    else:
        # Just load the pretrained model
        model = LitPCB(pretrained=True, batch_unpack_fn=pipeline_options.get('batch_unpack_fn'))

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


def run_custom_visualization_pipeline(pipeline_options: Dict):
    """
    Here I define a custom pipeline in order to reflect custom requirements for identities
    Possibly merge with the original version after changes there.

    """
    # 1. Initialize datamodule
    market_dm = Market1501DataModule(**pipeline_options['dataset_module_args'])
    market_dm.setup('test')

    features_paths = pipeline_options.get('features_paths')


    ###########
    visualizer = EmbeddingsVisualizer('./embeddings', market_dm.name)

    feat_storage_objects = {}
    fi = FeaturesInspector()
    for i, features_path in enumerate(features_paths):
        print(f'Processing {str(features_path)}')
        p = Path(features_path)
        if not p.exists():
            raise FileNotFoundError(features_path)

        # Get name from filepath
        model_name = p.stem

        # I need to explicitly specify the target_key because in the context of re-id the target (class label) is the id
        features_storage = FeaturesStorage(cached_path=str(p), target_key='id')

        features_storage.filter_by_ids(keep_range=(1, 15))

        # feats, labels = features_storage.raw_features()
        # fi.process('test', feats[1], labels[1])

        feat_storage_objects[model_name] = features_storage

    visualizer.add_features_multi(feat_storage_objects,
                                  datasets={'train': market_dm.dataset_gallery, 'test':market_dm.dataset_query})
    visualizer.plot(backend='bokeh', single_figure=False, images_server=pipeline_options.get('images_server'))


def run_custom_features_inspection_pipeline(pipeline_options: Dict):
    # 1. Initialize datamodule
    market_dm = Market1501DataModule(**pipeline_options['dataset_module_args'])
    market_dm.setup('test')

    features_paths = pipeline_options.get('features_paths')

    feat_storage_objects = {}
    fi = FeaturesInspector()
    for i, features_path in enumerate(features_paths):
        p = Path(features_path)
        print(f'Processing {p.name}...')
        if not p.exists():
            raise FileNotFoundError(features_path)

        # Get name from filepath
        model_name = p.stem

        # I need to explicitly specify the target_key because in the context of re-id the target (class label) is the id
        features_storage = FeaturesStorage(cached_path=str(p), target_key='id')

        features_storage.filter_by_ids(keep_range=(1, 15))

        # feats, labels = features_storage.raw_features()
        # fi.process('test', feats[1], labels[1])

        feat_storage_objects[model_name] = features_storage

        feats, labels = features_storage.raw_features()
        fi.process('test', feats[1], labels[1])


def save_epochs_feats(pipeline_options: Dict, weights_dir: str):
    weights_dir_path = Path(weights_dir)
    assert weights_dir_path.exists()

    for epoch_idx, p in enumerate(sorted(weights_dir_path.iterdir(), key=os.path.getmtime)):
        print(f'Running for checkpoint {p.name}')
        current_options = pipeline_options.copy()
        current_options['save_suffix'] = '{}_{}'.format(pipeline_options['save_suffix'], epoch_idx)
        current_options['weight_path'] = str(p)
        run_custom_testing_pipeline(pipeline_options=current_options, save_feats=True, eval=False)

dataset_name = 'market1501'

# batch_unpack_fn = lambda data: (data['image'], data['id'])

# fn_keys=('image', 'id', 'view_id', 'data_idx')
fn_keys = ('image', 'id', 'data_idx')
# fn_keys = ('image', 'id', 'view_id')

batch_unpack_fn = lambda batch_dict, keys=fn_keys: tuple(batch_dict[k] for k in keys)

common_options = \
    {
        'dataset_module_args':
            {
                'data_dir': '/media/amidemo/Data/reid_datasets/market1501/Market-1501-v15.09.15/',
                'batch_size': 64,
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
                        'batch_size': 64,
                        'num_instances': 4,
                        'batch_unpack_fn': batch_unpack_fn
                    }
            },
        'loss_class': PopulationAwareTripletLoss,
        'loss_args':
            {
                'loss_weight': 1.0,
                'loss_warm_up_epochs': 0,  # mute for re-id since we use pre-trained network and should be OK
                'semi_hard_warm_up_epochs': 0,  # mute for re-id since we use pre-trained network and should be OK
                'population_warm_up_epochs': 200  # put a large number (above the total epochs) to totally mute the population aware addition to the loss
            }
    }

run_training_pipeline(pipeline_options=merge(common_options, training_options))

#
# # Remember to set scale to original one in transforms
# weight_path_orig_scale = './lightning_logs/market1501_LitPCB/default/version_0/checkpoints/epoch=59-step=10995.ckpt'
weight_path_upscale = './lightning_logs/market1501_LitPCB/default/version_1/checkpoints/epoch=59-step=11093.ckpt'
# weight_poploss = './lightning_logs/market1501_LitPCB/default/version_2/checkpoints/epoch=51-step=9528.ckpt'
weight_path_upscale_pooling = './lightning_logs/market1501_LitPCB/default/version_3/checkpoints/epoch=59-step=10995.ckpt'
#
# weight_path_upscale_pooling_poploss = './lightning_logs/market1501_LitPCB/default/version_4/checkpoints/epoch=59-step=10995.ckpt'
weight_path_upscale_pooling_poploss = './lightning_logs/market1501_LitPCB/default/version_5/checkpoints/epoch=59-step=10995.ckpt'
weight_path_upscale_pooling_poploss_correct_bs = './lightning_logs/market1501_LitPCB/default/version_6/checkpoints/epoch=59-step=10995.ckpt'

testing_options = \
    {
        'weight_path': weight_path_upscale_pooling,
        'output_path': '/media/amidemo/Data/object_image_viz',
        'images_server': f'http://{platform.node()}:8345',
        'dataset_parts': ('test', ),
        'batch_unpack_fn': batch_unpack_fn,
        'save_suffix': '_orig_triloss'
    }

# run_custom_testing_pipeline(pipeline_options={**common_options, **testing_options}, save_feats=False, eval=True)

visualization_options = \
    {
        'features_paths':
            sorted(Path('./features/orig').iterdir(),
                   key=os.path.getmtime),
        'images_server': f'http://{platform.node()}:8345'
    }

# run_custom_visualization_pipeline(pipeline_options={**common_options, **visualization_options})
# run_custom_features_inspection_pipeline(pipeline_options={**common_options, **visualization_options})

# save_epochs_feats(pipeline_options={**common_options, **testing_options},
#                   weights_dir='./lightning_logs/market1501_LitPCB/lightning_logs/version_1/checkpoints')


# print(sorted(Path('./lightning_logs/market1501_LitPCB/lightning_logs/version_0/checkpoints').iterdir(),
#              key=os.path.getmtime))

# print(sorted(Path('./features/orig').iterdir(), key=os.path.getmtime))