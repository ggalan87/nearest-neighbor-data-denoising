import yaml
from itertools import product
from lightning.pipelines.pipelines import *

from lightning.cli_pipelines.common_options import bootstrap_datamodule, find_model_class_definition


from lightning.pipelines.utils import get_features_metric
import features_storage


def run_for_version_and_epoch(version: int, epoch: int, dataset_name: str, model_class_name: str):
    print(f'Extracting features for version {version}, epoch {epoch}')
    dm_name = dataset_name.lower()

    run_path = Path(f'./lightning_logs/{dm_name}_{model_class_name}/version_{version}')
    weight_path = run_path / f'checkpoints/epoch-{epoch}.ckpt'
    config_path = run_path / 'config.yaml'
    features_path = run_path / 'features'
    features_path.mkdir(exist_ok=True)

    config = yaml.load(open(config_path), yaml.SafeLoader)

    # I did it in try/catch block and not by using the get() method of the dict because in case it exists I want the
    # init_args field which would require extra expression
    try:
        dataset_args = config['data']['init_args']['dataset_args']['init_args']
    except KeyError:
        dataset_args = None

    dm = bootstrap_datamodule(dataset_name, sampler_args=None, dataset_args=dataset_args)

    model_class = find_model_class_definition(model_class_name)
    model = model_class.load_from_checkpoint(weight_path)

    dm.setup('test')
    gallery_dataloader, query_dataloader = dm.test_dataloader()

    storage = features_storage.FeaturesStorage(dataset_name=dm.name)

    # I use train/test notation for gallery/query since features storage was initially designed for classification
    # datasets which don't have such notation

    batch_keys = ('image', 'target', 'data_idx', 'is_noisy')
    storage.add('train', get_features_metric(dataloader=gallery_dataloader, model=model, batch_keys=batch_keys))
    storage.add('test', get_features_metric(dataloader=query_dataloader, model=model, batch_keys=batch_keys))
    storage.save(str(features_path / f'features_epoch-{epoch}.pt'))


dataset_name = 'ControlledPopulations'
model_class_name = 'ResNetMNIST'
versions = list(range(3, 5))
epochs = list(range(10))

for v, e in product(versions, epochs):
    run_for_version_and_epoch(v, e, dataset_name, model_class_name)
