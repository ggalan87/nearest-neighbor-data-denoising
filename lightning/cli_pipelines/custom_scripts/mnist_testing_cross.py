import yaml
from itertools import product
from lightning.pipelines.pipelines import *
from lightning.cli_pipelines.default_settings import *

from lightning.cli_pipelines.common_options import bootstrap_datamodule, find_model_class_definition


def run_for_version_and_epoch(version: int, epoch: int, model_dataset_name: str, model_class_name: str,
                              test_dataset_config_path: str):
    if not Path(test_dataset_config_path).exists():
        raise FileNotFoundError(test_dataset_config_path)

    print(f'Extracting features for version {version}, epoch {epoch} '
          f'using alternative dataset config from {test_dataset_config_path}')
    model_dm_name = model_dataset_name.lower()

    # Prepare the model path
    run_path = Path(f'./lightning_logs/{model_dm_name}_{model_class_name}/version_{version}')
    weight_path = run_path / f'checkpoints/epoch-{epoch}.ckpt'
    features_path = run_path / 'features_cross'
    features_path.mkdir(exist_ok=True)

    # Prepare the dataset
    test_config = yaml.load(open(test_dataset_config_path), yaml.SafeLoader)

    # I did it in try/catch block and not by using the get() method of the dict because in case it exists I want the
    # init_args field which would require extra expression
    try:
        dataset_args = test_config['init_args']['dataset_args']['init_args']
    except KeyError:
        dataset_args = None

    _, dm_name = test_config['class_path'].rsplit('.', 1)
    test_dataset_name = dm_name.replace('DataModule', '')
    dm = bootstrap_datamodule(test_dataset_name, sampler_args=None, dataset_args=dataset_args)

    model_class = find_model_class_definition(model_class_name)
    model = model_class.load_from_checkpoint(weight_path)
    extra_arguments =\
        {
            'batch_keys': ('image', 'target', 'data_idx'),
            'feat_parts': ('trainval', 'test'),
            'output_tag': f'features_{test_dataset_name}_epoch-{epoch}',
            'gpus': AVAIL_GPUS,
            'features_path': features_path
        }

    testing_pipeline(dm, model, extract_features=True, visualize=False, evaluate=False, eval_type='accuracy',
                     **extra_arguments)


dataset_name = 'MNIST'
model_class_name = 'ResNetMNIST'
versions = list(range(0, 1))
epochs = list(range(10))
test_dataset_config_path = '../configs/mnist/data_dirtymnist.yaml'

for v, e in product(versions, epochs):
    run_for_version_and_epoch(v, e, dataset_name, model_class_name, test_dataset_config_path)
