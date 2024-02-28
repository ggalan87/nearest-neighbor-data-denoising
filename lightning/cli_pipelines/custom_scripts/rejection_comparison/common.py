from pathlib import Path
from typing import Dict

import math
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryConfusionMatrix, BinaryF1Score

from torch_metric_learning.noise_reducers.sample_rejection.openset_helpers import NoiseHandlingOptions
from features_storage import FeaturesStorage

from lightning.cli_pipelines.common_options import bootstrap_datamodule
from torch_metric_learning.noise_reducers.sample_rejection import PopulationAwarePairRejection, \
    KNNPairRejection
from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import *


def to_list(t, batch_size=64):
    t_l = []

    n_samples = t.shape[0]
    n_full_batches = math.floor(n_samples / batch_size)
    for i in range(n_full_batches):
        t_l.append(t[i * batch_size:batch_size * (i + 1)])

    n_in_full = n_full_batches*batch_size
    t_l.append(t[n_in_full:n_in_full + (n_samples - n_in_full)])
    return t_l


def load_clean_training_labels(experiment_root: Path):
    assert experiment_root.exists()

    config_path = experiment_root / 'config.yaml'
    config = yaml.load(open(config_path), yaml.SafeLoader)

    # Quick and dirty solution for getting the dataset name without requiring it as an argument
    dataset_name = config['data']['class_path'].split('.')[-1].replace('DataModule', '')
    # I did it in try/catch block and not by using the get() method of the dict because in case it exists I want the
    # init_args field which would require extra expression
    try:
        dataset_args = config['data']['init_args']['dataset_args']['init_args']
    except (KeyError, TypeError) as e:
        dataset_args = None

    batch_size = 16  # config['data']['init_args']['batch_size']
    dm = bootstrap_datamodule(dataset_name, sampler_args=None, dataset_args=dataset_args, batch_size=batch_size)
    dm.setup('fit')

    training_dataset = dm.dataset_train.dataset
    training_df = pd.DataFrame(training_dataset.data)

    # return original labels and corresponding dataset indices
    return torch.tensor(training_df['target_orig']), torch.tensor(training_df.index)


def load_data(cached_path):
    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    # We entirely omit background images. In query images they do not exist but we do the same for completeness.
    (raw_training_feats, _), (raw_training_labels, _) = fs.raw_features()
    raw_training_indices = fs.training_feats['data_idx']
    raw_training_is_noisy = fs.training_feats['is_noisy']

    return raw_training_feats, raw_training_labels, raw_training_indices, raw_training_is_noisy


def load_data_full(cached_path, batch_size=0):
    # First load the data that were saved from feature extraction through the pass of the dataloader
    raw_training_feats, raw_training_labels, raw_training_indices, raw_training_is_noisy = load_data(cached_path)

    # Then load the true (clean) labels
    experiment_root = cached_path.parent.parent
    clean_training_labels, clean_training_indices = load_clean_training_labels(experiment_root)

    # Reorder the labels such that they follow the same order of the data from the dataloader ()
    clean_training_labels = clean_training_labels[raw_training_indices]

    # Check that we have loaded the data in correct order
    assert torch.all((clean_training_labels != raw_training_labels) == raw_training_is_noisy)

    if batch_size > 0:
        raw_training_feats = to_list(raw_training_feats, batch_size)
        raw_training_labels = to_list(raw_training_labels, batch_size)
        clean_training_labels = to_list(clean_training_labels, batch_size)
        raw_training_indices = to_list(raw_training_indices, batch_size)
        raw_training_is_noisy = to_list(raw_training_is_noisy, batch_size)

    return raw_training_feats, raw_training_labels, clean_training_labels, raw_training_indices, raw_training_is_noisy


def load_features(dataset_name, model_name, version, reference_epoch):
    dataset_name = dataset_name
    model_class_name = model_name
    dm_name = dataset_name.lower()

    training_epoch = reference_epoch + 1

    run_path = \
        Path(
            f'../../lightning_logs/{dm_name}_{model_class_name}/version_{version}')

    cached_path_reference = run_path / 'features' / f'features_epoch-{reference_epoch}.pt'
    cached_path_training = run_path / 'features' / f'features_epoch-{training_epoch}.pt'

    reference_feats, reference_labels, reference_indices, reference_is_noisy = load_data(cached_path_reference)

    feats, labels, indices, is_noisy = load_data(cached_path_training)

    return (reference_feats, reference_labels, reference_indices, reference_is_noisy), \
        (feats, labels, indices, is_noisy)


def find_sequential_paths(model_name, dataset_name, experiment_version, reference_epoch):
    model_class_name = model_name
    dm_name = dataset_name.lower()

    training_epoch = reference_epoch + 1

    run_path = \
        Path(
            f'../../lightning_logs/{dm_name}_{model_class_name}/version_{experiment_version}')

    cached_path_reference = run_path / 'features' / f'features_epoch-{reference_epoch}.pt'
    cached_path_training = run_path / 'features' / f'features_epoch-{training_epoch}.pt'

    return cached_path_reference, cached_path_training


def build_openset_param_string(openset_options):
    approach = openset_options['approach']
    distance_metric = openset_options['distance_metric']
    tail_size = openset_options['tail_size']

    if approach == 'EVM':
        cover_threshold = openset_options['cover_threshold']
        algorithm_parameters_string = \
            f'--distance_metric {distance_metric} --tailsize {tail_size} --cover_threshold {cover_threshold}'
    else:
        algorithm_parameters_string = \
            f'--distance_metric {distance_metric} --tailsize {tail_size}'

    return algorithm_parameters_string


def print_batch_stats(batch_predictions, batch_labels, predicted_as_noisy, batch_is_noisy):
    print(f'Found {torch.count_nonzero(torch.tensor(predicted_as_noisy))} noisy samples out of {len(batch_labels)}.')
    predicted_as_clean = torch.logical_not(torch.BoolTensor(predicted_as_noisy))
    gt_clean = torch.logical_not(batch_is_noisy)

    clean_weights = batch_predictions[batch_is_noisy == False]
    noisy_weights = batch_predictions[batch_is_noisy == True]

    # print(torch.mean(torch.median(clean_weights, dim=1)),
    #       torch.mean(torch.median(noisy_weights, dim=1)))

    clean_weights = []
    noisy_weights = []

    for i in range(len(batch_labels)):
        preds_i = batch_predictions[i]
        preds_i = preds_i[preds_i > 0.5]
        # print(bool(batch_is_noisy[i]), float(torch.mean(preds_i)))

        if batch_is_noisy[i]:
            noisy_weights.append(torch.mean(preds_i))
        else:
            clean_weights.append(torch.mean(preds_i))

    print(torch.mean(torch.tensor(clean_weights)) > torch.mean(torch.tensor(noisy_weights)))


def compute_global_stats(all_predicted_as_clean, all_gt_clean) -> Dict[str, torch.Tensor]:
    confusion_matrix_metric = BinaryConfusionMatrix(normalize='true')
    confusion_matrix = confusion_matrix_metric(all_predicted_as_clean, all_gt_clean)

    f1_score_metric = BinaryF1Score()
    f1_score = f1_score_metric(all_predicted_as_clean, all_gt_clean)

    return \
        {
            'confusion_matrix': confusion_matrix,
            'f1_score': f1_score
        }


def plot_batch_predictions(batch_predictions, batch_is_noisy):
    plt.figure()
    plt.matshow(batch_predictions[batch_is_noisy == False].numpy(), vmin=0, vmax=1)
    plt.colorbar()
    plt.title('noisy samples')
    plt.show()

    plt.figure()
    plt.matshow(batch_predictions[batch_is_noisy == True].numpy(), vmin=0, vmax=1)
    plt.colorbar()
    plt.title('clean samples')

    plt.show()
    #plt.savefig(f'{output_dir}/epoch-{epoch}-batch-{idx}.png')


def get_strategy(strategy, openset_options=None, other_options=None,
                 noise_handling_options=NoiseHandlingOptions(nn_weights=None, density_mr=None)):

    criterion = other_options['rejection_criterion']

    if strategy == 'openset':
        algorithm_parameters_string = build_openset_param_string(openset_options)
        rejection_strategy = \
            PopulationAwarePairRejection(approach=openset_options['approach'],
                                         algorithm_parameters_string=algorithm_parameters_string,
                                         training_samples_fraction=other_options['samples_fraction'],
                                         rejection_criteria=criterion,
                                         use_raw_probabilities=True,
                                         noise_handling_options=noise_handling_options)
        if noise_handling_options.inspection_callback is not None:
            # TODO:
            noise_handling_options.inspection_callback.set_rejection_strategy(rejection_strategy)

    elif strategy == 'knn':
        rejection_strategy = KNNPairRejection(k_neighbors=other_options['k_neighbors'],
                                              training_samples_fraction=other_options['samples_fraction'],
                                              rejection_criteria=criterion,
                                              weights_magnitude=1.0, use_raw_probabilities=False)
    else:
        raise NotImplementedError

    return rejection_strategy
