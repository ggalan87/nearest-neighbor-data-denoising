import yaml
from lightning.pipelines.pipelines import *

from lightning.cli_pipelines.common_options import bootstrap_datamodule


def prepare_paths(dataset_name, version, epoch):
    dm_name = dataset_name.lower()

    run_path = Path(f'./lightning_logs/{dm_name}_{model_class_name}/version_{version}')
    weight_path = run_path / f'checkpoints/epoch-{epoch}.ckpt'
    config_path = run_path / 'config.yaml'
    features_path = run_path / 'features'
    features_path.mkdir(exist_ok=True)
    output_path = run_path / 'embeddings'
    output_path.mkdir(exist_ok=True)

    return weight_path, config_path, features_path, output_path


def prepare_data(config_path: Path):
    config = yaml.load(open(config_path), yaml.SafeLoader)

    # I did it in try/catch block and not by using the get() method of the dict because in case it exists I want the
    # init_args field which would require extra expression
    try:
        dataset_args = config['data']['init_args']['dataset_args']['init_args']
    except KeyError:
        dataset_args = None

    dm = bootstrap_datamodule(dataset_name, sampler_args=None, dataset_args=dataset_args)
    individual_datasets = {}

    dm.setup(stage='test')
    individual_datasets['train'] = dm.dataset_gallery
    individual_datasets['test'] = dm.dataset_query
    return dm.name, individual_datasets


def run_for_version_and_epoch(version: int, epoch: int, dataset_name: str, model_class_name: str):
    weight_path, config_path, features_path, output_path = \
        prepare_paths(dataset_name, version, epoch)

    dm_name, datasets = prepare_data(config_path)

    feats_storage = features_storage.FeaturesStorage(cached_path=features_path / f'features_epoch-{epoch}.pt')

    visualizer = EmbeddingsVisualizer(output_path if output_path else './embeddings', dataset_name=dm_name)
    visualizer.add_features(f'{dm_name}_{model_class_name}_v{version}_e{epoch}', feats_storage, datasets=datasets)
    images_server = 'http://filolaos:8347'
    visualizer.plot('bokeh', images_server=images_server)


def run_aligned_for_versions_and_epoch(versions: List[int], epoch: int, dataset_name: str, model_class_name: str,
                                       output_path: str):
    features_dict = {}

    Path(output_path).mkdir(exist_ok=True)

    if len(versions) < 2:
        raise RuntimeError('At least two versions are required for computing aligned embeddings')

    for version in versions:
        weight_path, config_path, features_path, _ = \
            prepare_paths(dataset_name, version, epoch)

        dm_name, datasets = prepare_data(config_path)

        feats_storage = features_storage.FeaturesStorage(cached_path=features_path / f'features_epoch-{epoch}.pt')

        tag = f'{dm_name}_{model_class_name}_v{version}_e{epoch}'
        features_dict[tag] = feats_storage

    visualizer = EmbeddingsVisualizer(output_path if output_path else './embeddings', dataset_name=dm_name)
    visualizer.add_features_multi(features_dict=features_dict, datasets=datasets)
    images_server = 'http://filolaos:8347'
    visualizer.plot('bokeh', images_server=images_server)

dataset_name = 'ControlledPopulations'
model_class_name = 'LitModel'
versions = list(range(12, 16))
epochs = [0, 9]

# for v, e in product(versions, epochs):
#     run_for_version_and_epoch(version=v, epoch=e, dataset_name=dataset_name, model_class_name=model_class_name)

epoch = 9
aligned_output_path = f'/media/amidemo/Data/object_classifier_data/global_plots/controlledpopulations-hsv-plots/aligned_{epoch}'
run_aligned_for_versions_and_epoch(versions, epoch=epoch, dataset_name=dataset_name, model_class_name=model_class_name,
                                   output_path=aligned_output_path)
