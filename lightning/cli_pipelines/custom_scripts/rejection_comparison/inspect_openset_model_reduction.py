import logging
import pickle
import csv
from typing import Literal
from collections import defaultdict
from tqdm.contrib.itertools import product
from pathlib import Path

from evt.vast_openset import *
from torch.nn import functional as F

from torch_metric_learning.noise_reducers.memory_bank import MemoryBank
from torch_metric_learning.noise_reducers.sample_rejection.openset_helpers import NearestNeighborsWeightsOptions, \
    DensityBasedModelsReductionOptions, NoiseHandlingOptions
from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import HighScoreInPositiveClassCriterion, \
    KLDivergenceIsLowCriterion
from lightning.ext import logger
from common import load_data_full, find_sequential_paths, compute_global_stats, get_strategy


"""
This script examines various methods for addressing the issue that openset models are trained on noisy data

Two methods can be applied:
(a) Remove the noise before openset training
    (1) KNN
(b) Remove/ the noise after openset training
    (1) Model pruning
        (a) Density based
            (i) Sparsity threshold - needs ROC curve
            (ii) % Samples coverage
    (2) KNN prediction weighting

(c) Examine relabel in each case

It also presents the problem introduced with the noise
(a) results with clean data
(b) results with noisy-data and removed noise
(c) results with noisy-data

This must serve as a paradigm that noise harshly affects the openset model

Above results are in multiple directions:
(a) Classification
(b) # weibull models / e.g. for model pruning see if the correct models are pruned, how many correct samples vs noisy
samples remain after pruning, ratio per weibull model of clean/noisy
(c) for model
(d) Noise prediction (most important, however many other hyperparams are introduced)

ALSO:
I need to show that openset trained on clean data is better that simple KNN
"""


class WeibullModelsInspector:
    def __init__(self, noisy_labels, clean_labels):
        self._noisy_labels = noisy_labels
        self._clean_labels = clean_labels
        self._rejection_strategy = None
        self._data_where_reduced = False

    def set_rejection_strategy(self, rejection_strategy):
        self._rejection_strategy = rejection_strategy

    def _inspection_impl(self, trainer: OpensetTrainer):
        assert self._rejection_strategy is not None

        if not self._data_where_reduced:
            random_global_indices = self._rejection_strategy.random_global_indices

            self._noisy_labels = self._noisy_labels[random_global_indices]
            self._clean_labels = self._clean_labels[random_global_indices]
            self._data_where_reduced = True

        # Run for each class
        for label, model in trainer.models.items():
            class_features = trainer.data.features_dict[label]

            # The indices below where used for creating the class models
            class_indices = torch.where(self._noisy_labels == int(label))[0]
            class_noisy_mask = torch.ne(self._noisy_labels[class_indices], self._clean_labels[class_indices])

            n_noisy_per_weib = []
            for i, cv in enumerate(model['covered_vectors']):
                # ignore single-sample models
                if class_noisy_mask[cv].shape == torch.Size([]):
                    n_covered = 1
                else:
                    n_covered = len(class_noisy_mask[cv])

                n_noisy_per_weib.append(torch.count_nonzero(class_noisy_mask[cv]) / n_covered)
                # print(torch.count_nonzero(class_noisy_indices[cv]) / len(class_noisy_indices[cv])

            logger.debug(f'Label {label} noisy: {n_noisy_per_weib}')

            noisy_mask = class_noisy_mask[list(set(torch.hstack(model['covered_vectors']).tolist()))]
            n_global_noisy = torch.count_nonzero(noisy_mask)
            logger.info(f'Global noisy ratio for label {int(label)}: {n_global_noisy / len(noisy_mask)}')

    def __call__(self, trainer: OpensetTrainer):
        self._inspection_impl(trainer)


class MetricsOverTimeLogger:
    def __init__(self):
        self._metrics_history = defaultdict(list)
        self._history_length = 0

    def add_metrics(self, metrics: Dict[str, Any]):
        for name, value in metrics.items():
            self._metrics_history[name].append(value)
        self._history_length += 1

    def report(self):
        metrics_names = self._metrics_history.keys()
        for name in metrics_names:
            for t in range(self._history_length):
                print(f'{name} at {t}')
                print(self._metrics_history[name][t])

    def reset(self):
        self._metrics_history = defaultdict(list)
        self._history_length = 0

    @staticmethod
    def _confusion_matrix_headers():
        return ['TP', 'FP', 'FN', 'TN']

    @staticmethod
    def _confusion_matrix_raw_values(confusion_matrix: torch.Tensor):
        assert confusion_matrix.shape == (2, 2)

        return confusion_matrix.ravel().tolist()

    def export(self, filepath: str):
        export_path = Path(filepath)

        # Ensure that the parent folder exists
        assert export_path.parent.exists()

        file_ext = export_path.suffix

        if file_ext == '.pkl':
            with open(export_path, 'wb') as f:
                pickle.dump(self._metrics_history, f)
        elif file_ext == '.csv':
            with open(export_path, 'w', newline='') as f:

                field_names = []

                for name in self._metrics_history.keys():
                    if name == 'confusion_matrix':
                        field_names.extend(self._confusion_matrix_headers())
                    else:
                        field_names.append(name)

                writer = csv.DictWriter(f, fieldnames=field_names)
                writer.writeheader()

                for t in range(self._history_length):
                    row = {}
                    for name in self._metrics_history.keys():
                        if name == 'confusion_matrix':
                            raw_matrix = self._confusion_matrix_raw_values(self._metrics_history[name][t])
                            for column_name, value in zip(self._confusion_matrix_headers(), raw_matrix):
                                row[column_name] = value
                        else:
                            row[name] = self._metrics_history[name][t].item()
                    writer.writerow(row)
        else:
            raise NotImplementedError


def run_experiment(data_options, openset_options, other_options, noise_handling_options, global_strategy='openset'):
    metrics_logger = MetricsOverTimeLogger()

    cached_path_reference, cached_path_training = \
        find_sequential_paths(data_options['model_name'], data_options['dataset_name'], data_options['version'],
                              data_options['reference_epoch'])

    # Load all features
    reference_feats, reference_labels, reference_clean_labels, reference_indices, reference_is_noisy = \
        load_data_full(cached_path_reference)

    # Set here the inspection callback since it requires the noisy indices
    if noise_handling_options is not None:
        # noise_handling_options.inspection_callback = WeibullModelsInspector(noisy_labels=reference_labels,
        #                                                                     clean_labels=reference_clean_labels)
        pass

    rejection_strategy = get_strategy(global_strategy, openset_options, other_options, noise_handling_options)
    memory_bank = MemoryBank()

    if data_options['noisiness_type'] == 'noisy':
        epoch_labels = reference_labels
    elif data_options['noisiness_type'] == 'clean':
        epoch_labels = reference_clean_labels
    elif data_options['noisiness_type'] == 'clean_reduced':
        clean_indices = torch.where(torch.eq(reference_clean_labels, reference_labels))[0]
        reference_feats = reference_feats[clean_indices]
        epoch_labels = reference_labels[clean_indices]
        reference_indices = reference_indices[clean_indices]
    else:
        raise NotImplementedError

    memory_bank.add_to_memory(reference_feats, epoch_labels, dataset_indices=reference_indices)
    rejection_strategy.train(memory_bank)

    feats, labels, clean_labels, indices, is_noisy = load_data_full(cached_path_training, batch_size=64)
    all_noisy_predictions = []

    for idx, (batch_feats, batch_labels, batch_indices, batch_is_noisy, batch_clean_labels) in \
            enumerate(zip(feats, labels, indices, is_noisy, clean_labels)):
        batch_feats = F.normalize(batch_feats)

        batch_noise_predictions = rejection_strategy.predict_noise(batch_feats, batch_labels, normalize=False)
        all_noisy_predictions.extend(batch_noise_predictions)

        # print_batch_stats(inclusion_matrix, batch_labels, predicted_as_noisy, batch_is_noisy)
        #
        # if idx == 0:
        #     plot_batch_predictions(inclusion_matrix, batch_is_noisy)

    all_predicted_as_clean = torch.logical_not(torch.BoolTensor(all_noisy_predictions))
    all_gt_clean = torch.logical_not(torch.cat(is_noisy))

    stats = compute_global_stats(all_predicted_as_clean, all_gt_clean)
    return stats


def get_fixed_options():
    data_options = \
        {
            'dataset_name': 'Cars',
            'model_name': 'LitInception',
            'version': 282,
            'reference_epoch': 0,
            'batch_size': 64,
            'noisiness_type': 'noisy'  # noisy, clean, clean_reduced
        }

    openset_options = \
        {
            'tail_size': 0.1,
            'approach': 'EVM',
            'distance_metric': 'cosine',
            'cover_threshold': 0.7,
        }

    other_options = \
        {
            'k_neighbors': 200,
            'samples_fraction': 1.0,
            # 'density_strategy': 'coverage',
            # 'density_strategy_options':
            #     {
            #         'samples_coverage_ratio': 0.5
            #     }
            'density_strategy': 'threshold',
            'density_strategy_options':
                {
                    'score_threshold': 0.5
                },
            'rejection_criterion': HighScoreInPositiveClassCriterion(threshold=0.5)

        }

    return data_options, openset_options, other_options


def get_noise_handling_options(method: Literal['none', 'knn', 'density', 'knn+density'] = 'none',
                               method_options: Optional[Dict] = None):
    # Override none with an empty dictionary
    if method_options is None:
        method_options = {}

    if method == 'none':
        noise_handling_options = NoiseHandlingOptions(
            nn_weights=None,
            density_mr=None
        )
    elif method == 'knn':
        noise_handling_options = NoiseHandlingOptions(
            nn_weights=NearestNeighborsWeightsOptions(k_neighbors=method_options['k_neighbors'],
                                                      magnitude=method_options.get('magnitude', 1.0),
                                                      use_for_openset=method_options.get('use_for_openset', False)),
            density_mr=None
        )
    elif method == 'density':
        noise_handling_options = NoiseHandlingOptions(
            nn_weights=None,
            density_mr=DensityBasedModelsReductionOptions(strategy=method_options['density_strategy'],
                                                          strategy_options=method_options['density_strategy_options']),
            inspection_callback=None
        )
    elif method == 'knn+density':
        noise_handling_options = NoiseHandlingOptions(
            nn_weights=NearestNeighborsWeightsOptions(k_neighbors=method_options['k_neighbors'],
                                                      magnitude=method_options.get('magnitude', 1.0),
                                                      use_for_openset=method_options.get('use_for_openset', False)),
            density_mr=DensityBasedModelsReductionOptions(strategy=method_options['density_strategy'],
                                                          strategy_options=method_options['density_strategy_options'])

        )
    else:
        raise NotImplementedError

    return noise_handling_options


def before_training(options=None):
    data_options, openset_options, other_options = options if options is not None else get_fixed_options()

    # This is true to run knn before training openset, and thus use knn-filtered data
    other_options['use_for_openset'] = True
    # The zero below mutes the knn-based inclusion probability re-weighting, thus knn is used only before openset to
    # reduce the data and not afterwards
    other_options['magnitude'] = 0.0

    stats = run_experiment(data_options, openset_options, other_options,
                   get_noise_handling_options(method='knn', method_options=other_options))
    print(stats)
    return stats


def after_training(method: Literal['pruning', 'weighting'], options=None):
    data_options, openset_options, other_options = options if options is not None else get_fixed_options()

    if method == 'pruning':
        # (1) Model pruning
        stats = run_experiment(data_options, openset_options, other_options,
                       get_noise_handling_options(method='density', method_options=other_options))
    elif method == 'weighting':
        # (2) KNN weighting
        stats = run_experiment(data_options, openset_options, other_options,
                       get_noise_handling_options(method='knn', method_options=other_options))
    else:
        raise NotImplementedError

    return stats


def run_knn_global_strategy():
    data_options, openset_options, other_options = get_fixed_options()
    # run_experiment(data_options, openset_options, other_options,
    #                get_noise_handling_options(method='none'), global_strategy='knn')
    stats = run_experiment(data_options, openset_options, other_options,
                           get_noise_handling_options(method='none'), global_strategy='knn')
    print(stats)


def run_knn_global_strategy_ablation():
    data_options, openset_options, other_options = get_fixed_options()

    reference_epochs = torch.arange(0, 29, 1)
    neighbors = [1, 2, 5, 10, 20, 50, 100, 200, 400, 600, 1000]
    samples_fraction = [1.0]
    criteria_list = \
        [
            HighScoreInPositiveClassCriterion(threshold=0.5),
            # KLDivergenceIsLowCriterion(threshold=t) for t in [0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995]
        ]

    for hparams in product(neighbors, samples_fraction, criteria_list):
        k, sf, criterion = hparams

        metrics_logger = MetricsOverTimeLogger()

        for re in reference_epochs:
            print(f'Reference epoch {int(re)}, k: {k}, data_ratio: {sf}')
            data_options['reference_epoch'] = int(re)
            other_options['k_neighbors'] = k
            other_options['samples_fraction'] = sf
            other_options['rejection_criterion'] = criterion
            stats = run_experiment(data_options, openset_options, other_options,
                                   get_noise_handling_options(method='none'), global_strategy='knn')

            metrics_logger.add_metrics(stats)

        metrics_logger.export(f'knn_global_strategy_{str(hparams)}.csv')


def run_problem_statement():
    data_options, openset_options, other_options = get_fixed_options()

    # (a) results with clean data
    data_options['noisiness_type'] = 'clean'
    none_method(options=(data_options, openset_options, other_options))

    # (b) results with noisy-data and removed noise
    data_options['noisiness_type'] = 'clean_reduced'
    none_method(options=(data_options, openset_options, other_options))

    # (c) results with noisy-data
    data_options['noisiness_type'] = 'noisy'
    none_method(options=(data_options, openset_options, other_options))


def none_method(options=None):
    """
    Run openset without any method for addressing the noisy samples
    """
    data_options, openset_options, other_options = options if options is not None else get_fixed_options()
    run_experiment(data_options, openset_options, other_options, get_noise_handling_options(method='none'))


def run_experiments():
    none_method()
    before_training()
    after_training(method='pruning')
    after_training(method='weighting')


def run_after_training_pruning():
    metrics_logger = MetricsOverTimeLogger()

    data_options, openset_options, other_options = get_fixed_options()
    openset_options['tail_size'] = 1.0
    other_options['density_strategy'] = 'threshold'
    other_options['density_strategy_options'] = {'score_threshold': 0.1}
    stats = after_training(method='pruning', options=(data_options, openset_options, other_options))

    print(stats)
    # metrics_logger.add_metrics(stats)

    # metrics_logger.export(f'density_based_pruning.csv')


def run_before_training_ablation():
    data_options, openset_options, other_options = get_fixed_options()

    reference_epochs = torch.arange(0, 29, 1)

    versions = [0, 1]
    tail_sizes = [0.1, 0.5, 0.9]

    for hparams in product(versions, tail_sizes):
        version, tail_size = hparams

        data_options['version'] = version
        openset_options['tail_size'] = tail_size

        metrics_logger = MetricsOverTimeLogger()

        for re in reference_epochs:
            print(f'Reference epoch {int(re)}, version: {version}')
            data_options['reference_epoch'] = int(re)
            stats = before_training((data_options, openset_options, other_options))

            metrics_logger.add_metrics(stats)

        metrics_logger.export(f'before_training_{str(hparams)}.csv')


def run_threshold_ablation():
    data_options, openset_options, other_options = get_fixed_options()

    other_options['density_strategy'] = 'threshold'

    thresholds = torch.hstack((torch.arange(0.0, 0.1, 0.01), torch.arange(0.1, 1.0, 0.1)))

    for t in thresholds:
        print(f'Threshold {t.item()}')
        other_options['density_strategy_options'] = {'score_threshold': float(t)}
        after_training(method='pruning', options=(data_options, openset_options, other_options))


def run_clean_data_ts_ablation():
    data_options, openset_options, other_options = get_fixed_options()

    data_options['noisiness_type'] = 'clean'

    tail_sizes = torch.arange(0.1, 1.1, 0.1)
    for t in tail_sizes:
        print(f'TS {t.item()}')
        openset_options['tail_size'] = float(t)
        none_method(options=(data_options, openset_options, other_options))


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    run_knn_global_strategy_ablation()
    # run_knn_global_strategy()
    # run_experiments()
    # run_before_training_ablation()
    # run_threshold_ablation()
    # run_problem_statement()
    # run_clean_data_ts_ablation()
    # run_after_training_pruning()
