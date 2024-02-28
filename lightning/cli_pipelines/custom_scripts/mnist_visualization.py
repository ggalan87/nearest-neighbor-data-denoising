import yaml
from lightning.pipelines.pipelines import *

from lightning.cli_pipelines.common_options import bootstrap_datamodule


def run_for_version_and_epoch(version: int, epoch: int, dataset_name: str, model_class_name: str):
    dm_name = dataset_name.lower()

    run_path = Path(f'./lightning_logs/{dm_name}_{model_class_name}/version_{version}')
    weight_path = run_path / f'checkpoints/epoch-{epoch}.ckpt'
    config_path = run_path / 'config.yaml'
    features_path = run_path / 'features'
    features_path.mkdir(exist_ok=True)
    output_path = run_path / 'embeddings'
    output_path.mkdir(exist_ok=True)

    config = yaml.load(open(config_path), yaml.SafeLoader)

    # I did it in try/catch block and not by using the get() method of the dict because in case it exists I want the
    # init_args field which would require extra expression
    try:
        dataset_args = config['data']['init_args']['dataset_args']['init_args']
    except KeyError:
        dataset_args = None

    dm = bootstrap_datamodule(dataset_name, sampler_args=None, dataset_args=dataset_args)

    fs = features_storage.FeaturesStorage(cached_path=features_path / f'features_epoch-{epoch}.pt')
    features_tag = f'{dm.name}_{model_class_name}_v{version}_e{epoch}'

    images_server = 'http://filolaos:8347'
    visualization_pipeline(dm, feats_storage=fs, features_tag=features_tag, feat_parts=('trainval', 'test'),
                           output_path=str(output_path), images_server=images_server)


dataset_name = 'MNISTSubset'
model_class_name = 'LitModel'
versions = list(range(0, 1))
epochs = list(range(10))

for v in [2]:
    for e in [0, 9]:
        run_for_version_and_epoch(version=v, epoch=e, dataset_name=dataset_name, model_class_name=model_class_name)
