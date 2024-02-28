import yaml
from itertools import product
from lightning.pipelines.pipelines import *
from lightning.cli_pipelines.default_settings import *

from lightning.cli_pipelines.common_options import bootstrap_datamodule, find_model_class_definition


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
    extra_arguments =\
        {
            'batch_keys': ('image', 'target', 'data_idx'),
            # 'batch_keys': ('image', 'target', 'data_idx', 'is_noisy'),
            'feat_parts': ('trainval', 'test'),
            'output_tag': f'features_epoch-{epoch}',
            'gpus': AVAIL_GPUS,
            'features_path': features_path
        }

    testing_pipeline(dm, model, extract_features=True, visualize=False, evaluate=False, eval_type='accuracy',
                     **extra_arguments)


dataset_name = 'MNIST'
model_class_name = 'LitResnet'
versions = list(range(2, 3))
epochs = list(range(10))

for v, e in product(versions, epochs):
    run_for_version_and_epoch(v, e, dataset_name, model_class_name)
