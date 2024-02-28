import argparse
import yaml
from pathlib import Path
from itertools import product
from typing import List
from lightning.pipelines.pipelines import testing_pipeline
from lightning.cli_pipelines.default_settings import *

from lightning.cli_pipelines.common_options import bootstrap_datamodule, find_model_class_definition


def run_for_version_and_epoch(version: int, epoch: int, dataset_name: str, target_dataset_name: str,
                              model_class_name: str, parts: List[str],
                              batch_keys: List[str]):
    dm_name = dataset_name.lower()

    run_path = Path(f'./lightning_logs/{dm_name}_{model_class_name}/version_{version}')
    weight_path = run_path / f'checkpoints/epoch-{epoch}.ckpt'

    # -1 is a special case for extracting features from the pretrained model
    if not weight_path.exists() and epoch != -1:
        return

    config_path = run_path / 'config.yaml'
    features_path = run_path / (
        'features' if target_dataset_name is None else f'features-{target_dataset_name.lower()}')
    features_path.mkdir(exist_ok=True)

    config = yaml.load(open(config_path), yaml.SafeLoader)

    # I did it in try/catch block and not by using the get() method of the dict because in case it exists I want the
    # init_args field which would require extra expression
    try:
        dataset_args = config['data']['init_args']['dataset_args']['init_args']
    except (KeyError, TypeError) as e:
        dataset_args = None

    if target_dataset_name is not None:
        dataset_name = target_dataset_name

    batch_size = config['data']['init_args']['batch_size']
    dm = bootstrap_datamodule(dataset_name, sampler_args=None, dataset_args=dataset_args, batch_size=batch_size)

    model_class = find_model_class_definition(model_class_name)

    if epoch != -1:
        model = model_class.load_from_checkpoint(weight_path)
    else:
        model_init_args = config['model']['init_args']
        # override only the batch size
        model_init_args['batch_size'] = batch_size

        # Link to datamodule num classes
        # (most probably not required, since we simply extract features and not class probabilities)
        model_init_args['num_classes'] = dm.num_classes

        # Prevent training phase kwargs from initialization., Some of them should be converted from strings to classes,
        # otherwise initialization of them does not affect the process
        for k in list(model_init_args.keys()):
            if '_kwargs' in k or '_class' in k:
                del model_init_args[k]

        if not model_init_args['use_pretrained_weights']:
            raise AssertionError('Model wan not trained using pretrained weights, while requested extracting them.')

        model = model_class(**model_init_args)

    # TODO: extract additional tags if available, e.g. is_noisy etc
    extra_arguments = \
        {
            'batch_keys': tuple(batch_keys),
            'feat_parts': parts,
            'output_tag': f'features_epoch-{epoch}',
            'gpus': AVAIL_GPUS,
            'features_path': features_path
        }

    print(f'Extracting features for version {version}, epoch {epoch} from dataset {dataset_name}')

    testing_pipeline(dm, model, extract_features=True, visualize=False, evaluate=False, eval_type='accuracy',
                     **extra_arguments)


def run(args):
    dataset_name = args.dataset_name
    target_dataset_name = args.target_dataset_name
    model_class_name = args.model_name

    if args.versions_range is not None:
        versions = list(range(args.versions_range[0], args.versions_range[1] + 1))
    else:
        versions = args.versions_list

    if args.epochs_range is not None:
        epochs = list(range(args.epochs_range[0], args.epochs_range[1] + 1))
    else:
        epochs = args.epochs_list

    for v, e in product(versions, epochs):
        run_for_version_and_epoch(v, e, dataset_name, target_dataset_name,
                                  model_class_name, args.parts_list, args.batch_keys)


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-name', required=True, type=str,
                        help='The name of the dataset, or more precisely the class name.')
    parser.add_argument('--target-dataset-name', required=False, type=str,
                        help='The name of the dataset from which to extract features, or more precisely the class name.'
                             'If not specified it will be the same as dataset-name arg.')
    parser.add_argument('--model-name', required=True, type=str,
                        help='The name of the model, or more precisely the model class name.')
    parser.add_argument('--parts-list', nargs='+', type=str, default=['test'],
                        help='Dataset part(s) for which to compute features')
    parser.add_argument('--batch-keys', nargs='+', type=str, default=['image', 'target', 'data_idx'],
                        help='Which dataset keys to get and save. Others include e.g. img path etc')

    versions_group = parser.add_mutually_exclusive_group(required=True)
    versions_group.add_argument('--versions-list', nargs='+', type=int, help='List of versions')
    versions_group.add_argument('--versions-range', nargs=2, type=int, help='Range of versions (min,max included)')

    epochs_group = parser.add_mutually_exclusive_group(required=True)
    epochs_group.add_argument('--epochs-list', nargs='+', type=int, help='List of epochs')
    epochs_group.add_argument('--epochs-range', nargs=2, type=int, help='Range of epochs (min,max included)')

    return parser.parse_args()


if __name__ == '__main__':
    run(parse_cli())
