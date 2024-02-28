from pathlib import Path
import importlib
from typing import Literal, Dict
import yaml
import torch
import torch.nn.functional as F
from vast_openset import OpensetData, OpensetModelParameters, OpensetTrainer
from features_storage import FeaturesStorage

from lightning.pipelines_datasets_scripts.mnist.run import CustomCLI


def get_config_from_cli(config_path: Path):
    args = [f'--config={config_path}']
    cli = CustomCLI(args=args, run=False)
    return cli.config


def fix_for_dirty(config_path: Path, raw_testing_labels: torch.Tensor, raw_testing_indices: torch.Tensor)\
        -> torch.Tensor:
    def get_class(class_path: str):
        module_name, class_name = class_path.rsplit('.', 1)
        return getattr(importlib.import_module(module_name), class_name)

    config = yaml.load(open(config_path), yaml.SafeLoader)
    data_config = config['data']
    if 'Dirty' not in data_config['class_path']:
        return raw_testing_labels

    datamodule_class_ = get_class(data_config['class_path'])
    dataset_args_class_ = get_class(data_config['init_args']['dataset_args']['class_path'])
    data_config['init_args']['dataset_args'] = \
        dataset_args_class_(**data_config['init_args']['dataset_args']['init_args'])
    datamodule = datamodule_class_(**data_config['init_args'])
    datamodule.setup('test')
    dirty_indices = torch.load(datamodule.dataset_test.cached_filepath_indices)

    # assume testing indices are in ascending order because no sampling or randomization has been applied
    assert list(raw_testing_indices.numpy()) == list(range(0, len(raw_testing_indices)))

    raw_testing_labels[dirty_indices == 1] = raw_testing_labels.max() + 1
    return raw_testing_labels


def load_data_from_experiment(experiment_root: Path, epoch=0, features_template='features/features_epoch-{}.pt',
                              alternative_config_path: Path = None, training_data_ratio=1.0):
    assert experiment_root.exists()

    features_path = experiment_root / features_template.format(epoch)
    assert features_path.exists()

    fs = FeaturesStorage(cached_path=features_path, target_key='target')

    (raw_training_feats, raw_testing_feats), (raw_training_labels, raw_testing_labels) = fs.raw_features()
    raw_training_indices, raw_testing_indices = fs.training_feats['data_idx'], fs.testing_feats['data_idx']

    config_path = experiment_root / 'config.yaml' if alternative_config_path is None else alternative_config_path
    raw_testing_labels = \
        fix_for_dirty(config_path, raw_testing_labels=raw_testing_labels, raw_testing_indices=raw_testing_indices)

    print(f'Experiment with {len(torch.unique(raw_training_labels))} classes')

    raw_training_feats = F.normalize(raw_training_feats)
    raw_testing_feats = F.normalize(raw_testing_feats)

    if training_data_ratio < 1.0:
        random_indices =\
            torch.randperm(len(raw_training_labels))[:int(training_data_ratio* len(raw_training_labels))]
        raw_training_feats = raw_training_feats[random_indices]
        raw_training_labels = raw_training_labels[random_indices]

    training_data = OpensetData(raw_training_feats, raw_training_labels, None)
    testing_data = OpensetData(raw_testing_feats, raw_testing_labels, None)

    return training_data, testing_data


example_configurations = \
    {
        'OpenMax':
            [
                {'distance_metric': 'euclidean', 'tailsize': 1.0},
                {'distance_metric': 'euclidean', 'tailsize': 0.5},
                {'distance_metric': 'euclidean', 'tailsize': 0.25},
                {'distance_metric': 'euclidean', 'tailsize': 0.1},
            ],
        'EVM':
            [
                {'distance_metric': 'euclidean', 'tailsize': 1.0},
                {'distance_metric': 'euclidean', 'tailsize': 1.0, 'distance_multiplier': 0.7},
                {'distance_metric': 'euclidean', 'tailsize': 0.5},
                {'distance_metric': 'euclidean', 'tailsize': 0.25},
                {'distance_metric': 'euclidean', 'tailsize': 0.1}
            ],
        'MultiModalOpenMax':
            [
                {'distance_metric': 'euclidean', 'tailsize': 1},
                {'distance_metric': 'euclidean', 'tailsize': 1, 'Clustering_Algo': 'finch'},
            ]
    }


class VastOpensetPipelineParams:
    def __init__(self, approach: Literal['OpenMax', 'EVM', 'MultiModalOpenMax'], parameters: Dict, save=False):
        # create parameter string
        algo_params_string = ''
        for k, v in parameters.items():
            algo_params_string += f'--{k} {v} '
        algo_params_string = algo_params_string.strip()

        saver_parameters_string = f"--OOD_Algo {approach}" if save else None

        self._model_params = OpensetModelParameters(approach, algo_params_string, saver_parameters_string)

    @property
    def model_params(self):
        return self._model_params


experiment_path = '../lightning/pipelines_datasets_scripts/mnist/lightning_logs/mnist_ResNetMNIST/version_0'
alternative_config_path = \
    '../lightning/pipelines_datasets_scripts/mnist/lightning_logs/dirtymnist_ResNetMNIST/version_7/config.yaml'

approach = 'EVM'

if approach == 'EVM':
    training_data_ratio = 0.3

training_data, _ = load_data_from_experiment(Path(experiment_path), epoch=9, training_data_ratio=training_data_ratio)
_, testing_data = load_data_from_experiment(Path(experiment_path), epoch=9,
                                            features_template='features_cross/features_DirtyMNIST_epoch-{}.pt',
                                            alternative_config_path=Path(alternative_config_path),
                                            training_data_ratio=training_data_ratio)


for example_config in example_configurations[approach][2:4]:
    trainer = OpensetTrainer(training_data, VastOpensetPipelineParams(approach, example_config).model_params)
    trainer.train()
    trainer.eval(testing_data)
