from tqdm.contrib.itertools import product

from evt.vast_openset import *
from torch.nn import functional as F

from torch_metric_learning.noise_reducers.memory_bank import MemoryBank
from torch_metric_learning.noise_reducers.sample_rejection import PopulationAwarePairRejection, KNNPairRejection
from torch_metric_learning.noise_reducers.sample_rejection.openset_helpers import NoiseHandlingOptions
from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import HighScoreInPositiveClassCriterion, \
    CombinedCriteria

from common import load_data_full, find_sequential_paths, build_openset_param_string  # , print_global_stats


def simulate_noisy_rejection(dataset_name, model_name, version, reference_epoch, batch_size, openset_options,
                             weights_magnitude=1.0, k_neighbors=20, strategy='openset', samples_fraction=1.0,
                             train_on_clean=False):

    cached_path_reference, cached_path_training =\
        find_sequential_paths(model_name, dataset_name, version, reference_epoch)

    # We load all data / fraction = 1.0
    reference_feats, reference_labels, reference_clean_labels, reference_indices, reference_is_noisy = \
        load_data_full(cached_path_reference)

    algorithm_parameters_string = build_openset_param_string(openset_options)

    if strategy == 'openset':
        noise_handling_options = NoiseHandlingOptions(
            nn_weights=None,
            density_mr=None
        )
        # noise_handling_options = NoiseHandlingOptions(
        #     nn_weights=NearestNeighborsWeightsOptions(k_neighbors=k_neighbors, magnitude=weights_magnitude, use_for_openset=False),
        #     density_mr=None
        # )
        # noise_handling_options = NoiseHandlingOptions(
        #     nn_weights=None,
        #     density_mr=DensityBasedModelsReductionOptions(strategy_options={'samples_coverage_ratio': 0.5})
        # )
        rejection_strategy = \
            PopulationAwarePairRejection(approach=openset_options['approach'],
                                         algorithm_parameters_string=algorithm_parameters_string,
                                         training_samples_fraction=samples_fraction,
                                         use_raw_probabilities=True,
                                         noise_handling_options=noise_handling_options)
    elif strategy == 'knn':
        rejection_strategy = KNNPairRejection(k_neighbors=k_neighbors, training_samples_fraction=samples_fraction,
                                              weights_magnitude=weights_magnitude)
    else:
        raise NotImplementedError

    memory_bank = MemoryBank()
    memory_bank.add_to_memory(reference_feats, reference_clean_labels if train_on_clean else reference_labels)
    rejection_strategy.train(memory_bank)

    feats, labels, clean_labels, indices, is_noisy = load_data_full(cached_path_training, batch_size=64)

    # testing_data = OpensetData(features=feats, class_labels=labels)
    # rejection_strategy._openset_trainer.eval_classification(testing_data)

    all_noisy_predictions = []
    for idx, (batch_feats, batch_labels, batch_indices, batch_is_noisy, batch_clean_labels) in \
            enumerate(zip(feats, labels, indices, is_noisy, clean_labels)):

        batch_feats = F.normalize(batch_feats)

        rejection_strategy.predict_noise(batch_feats, batch_labels, normalize=False)

        # Inclusion probabilities - NxC
        inclusion_matrix = rejection_strategy.current_batch_noisy_predictions

        # Labels are not guaranteed to be in 0..C-1 range, and therefore we need a "re-map" for fast indexing
        labels_to_indices = rejection_strategy.labels_to_indices
        label_idx = torch.tensor([labels_to_indices[int(label)] for label in batch_labels])

        criteria = \
            [
                HighScoreInPositiveClassCriterion(threshold=0.5),
                # LowScoreInNegativeCDon LancasterlassCriterion(threshold=0.5),
                # HighestIsSoloCriterion(threshold=0.1)
            ]

        criteria = CombinedCriteria(criteria)
        predicted_as_noisy = criteria.sample_is_noisy(label_idx, inclusion_matrix)
        all_noisy_predictions.extend(predicted_as_noisy)

        # print_batch_stats(inclusion_matrix, batch_labels, predicted_as_noisy, batch_is_noisy)
        #
        # if idx == 0:
        #     plot_batch_predictions(inclusion_matrix, batch_is_noisy)

    all_predicted_as_clean = torch.logical_not(torch.BoolTensor(all_noisy_predictions))
    all_gt_clean = torch.logical_not(torch.cat(is_noisy))

    # print_global_stats(all_predicted_as_clean, all_gt_clean)


def run(dataset_name, model_name, version, reference_epoch, batch_size=64, samples_fraction=1.0):
    tail_sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # distance_metrics = ['euclidean', 'cosine']
    # magnitudes = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    # neighbors = [5, 10, 20, 50, 100, 200]

    # tail_sizes = [0.9]
    distance_metrics = ['cosine']
    magnitudes = [1.0]
    neighbors = [200]

    results = []
    for tail_size, distance_metric, magnitude, k in product(tail_sizes, distance_metrics, magnitudes, neighbors):
        openset_options = \
            {
                'tail_size': tail_size,
                'reduced_dim': -1,
                'approach': 'EVM',
                'distance_metric': distance_metric,
                'cover_threshold': 0.7,
            }

        confusion_matrix = \
            simulate_noisy_rejection(dataset_name, model_name, version, reference_epoch, batch_size=batch_size,
                                     openset_options=openset_options, weights_magnitude=magnitude, k_neighbors=k,
                                     strategy='openset', samples_fraction=samples_fraction, train_on_clean=True)
        config_and_result = {**openset_options,
                             **{'weights_magnitude': magnitude, 'k_neighbors': k, 'confusion_matrix': confusion_matrix}}
        results.append(config_and_result)
    #
    # with open('configs_and_results_cars.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # results = []
    # for magnitude, k in product(magnitudes, neighbors):
    #     openset_options = \
    #         {
    #             'tail_size': tail_sizes[0],
    #             'reduced_dim': -1,
    #             'approach': 'EVM',
    #             'distance_metric': distance_metrics[0],
    #             'cover_threshold': 0.7
    #         }
    #     confusion_matrix = \
    #         simulate_noisy_rejection(dataset_name, model_name, version, reference_epoch, batch_size=batch_size,
    #                                  openset_options=openset_options, weights_magnitude=magnitude, k_neighbors=k,
    #                                  strategy='knn')
    #     config_and_result = {'weights_magnitude': magnitude, 'k_neighbors': k, 'confusion_matrix': confusion_matrix}
    #     results.append(config_and_result)
    #
    # with open('configs_and_results_knn_cars.pkl', 'wb') as f:
    #     pickle.dump(results, f)
