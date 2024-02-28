from evt.vast_openset import *

from pytorch_lightning.utilities.seed import seed_everything

from torch_metric_learning.noise_reducers.sample_rejection import PopulationAwarePairRejection, KNNPairRejection
from torch_metric_learning.noise_reducers.sample_rejection.openset_helpers import NearestNeighborsWeightsOptions
import pandas as pd

from lightning.data.dataset_utils import disturb_targets_symmetric

import torch
from torchmetrics.classification import BinaryConfusionMatrix

from pytorch_metric_learning.losses import CrossBatchMemory
from pytorch_metric_learning.losses import ContrastiveLoss

from torch_metric_learning.miners.prism import PRISM
from torch_metric_learning.noise_reducers.sample_rejection.openset_helpers import NoiseHandlingOptions


# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7932895


def load_data(train=True):
    if train:
        data = pd.read_csv('../../../../evt/TestData/train_mnist.csv').to_numpy()
    else:
        data = pd.read_csv('../../../../evt/TestData/test_mnist.csv').to_numpy()

    features = torch.Tensor(data[:, 1:3])
    labels = torch.IntTensor(data[:, 0])

    if train:
        random_indices = torch.randperm(len(labels))[:int(0.3 * len(labels))]
        features = features[random_indices]
        labels = labels[random_indices]

    return features.contiguous(), labels


def raw_openset(approach, algorithm_parameters_string, training_features: torch.Tensor, training_labels: torch.Tensor,
                noisy_training_labels: torch.Tensor, testing_features: torch.Tensor, testing_labels: torch.Tensor,
                noisy_testing_labels: torch.Tensor):
    training_data = OpensetData(features=training_features, class_labels=training_labels)
    noisy_training_data = OpensetData(features=training_features, class_labels=noisy_training_labels)
    testing_data = OpensetData(features=testing_features, class_labels=testing_labels)

    saver_parameters = f"--OOD_Algo {approach}"

    noise_handling_options = NoiseHandlingOptions(
        nn_weights=NearestNeighborsWeightsOptions(k_neighbors=200, magnitude=1.2, use_for_openset=False),
        density_mr=None
    )
    model_params = OpensetModelParameters(approach, algorithm_parameters_string, saver_parameters)

    print('REGULAR')
    for i in range(1):
        trainer = OpensetTrainer(training_data, model_params, noise_handling_options=None)
        trainer.train()
        # trainer.eval_classification(testing_data)
        trainer.plot(testing_data)

        print({ev: len(trainer._models[ev]['extreme_vectors']) for ev in trainer._models.keys()})

    print('NOISY')
    for i in range(1):
        trainer = OpensetTrainer(noisy_training_data, model_params, noise_handling_options=noise_handling_options,
                                 verbose=True)
        trainer.train()
        # trainer.eval_classification(testing_data)
        # trainer.plot(testing_data)
        #
        # for ev in trainer._models.keys():
        #     print(len(trainer._models[ev]['extreme_vectors']))


def with_rejection_strategy(approach, algorithm_parameters_string, training_features: torch.Tensor,
                            training_labels: torch.Tensor, noisy_training_labels: torch.Tensor,
                            testing_features: torch.Tensor, testing_labels: torch.Tensor,
                            noisy_testing_labels: torch.Tensor):

    rejection_strategy = \
        PopulationAwarePairRejection(approach=approach, algorithm_parameters_string=algorithm_parameters_string,
                                     training_samples_fraction=1.0, weighted_openset_probabilities=True)
    rejection_strategy = KNNPairRejection(k_neighbors=20, training_samples_fraction=1.0)
    rejection_strategy.keep_epoch_features(training_features, noisy_training_labels)
    rejection_strategy.train()

    #testing_labels[noisy_testing_labels != testing_labels] = len(rejection_strategy.labels_to_indices) + 1
    # _, _, metrics = rejection_strategy._openset_trainer.eval(OpensetData(testing_features, testing_labels, None, None),
    #                                                          get_metrics=True)

    rejection_strategy.predict_noise(testing_features)
    inclusion_matrix = rejection_strategy.retrieve_batch_predictions()

    # Labels are not guaranteed to be in 0..C-1 range, and therefore we need a "re-map" for fast indexing
    labels_to_indices = rejection_strategy.labels_to_indices

    predicted_as_noisy = []
    for i in range(len(noisy_testing_labels)):
        inclusion_vec = inclusion_matrix[i]

        # print(inclusion_vec)

        # (A)
        # if the sample is not included in its corresponding class, we regard it as noisy

        crit1_noisy = False
        crit2_noisy = False

        if inclusion_vec[labels_to_indices[int(noisy_testing_labels[i])]] == 0:
            crit1_noisy = True

        # create a mask initialized by all True
        selected_indices = torch.ones((inclusion_vec.shape[0],), dtype=torch.bool)
        # exclude same class index
        selected_indices[labels_to_indices[int(noisy_testing_labels[i])]] = False
        # check if inclusion label of the sample corresponds to another class and if so, exclude the
        # corresponding samples as described above
        if any(inclusion_vec[selected_indices]):
            crit2_noisy = False

        is_noisy = crit1_noisy or crit2_noisy
        # print(is_noisy)
        predicted_as_noisy.append(is_noisy)

    predicted_as_clean = torch.logical_not(torch.BoolTensor(predicted_as_noisy))
    gt_clean = noisy_testing_labels == testing_labels

    confusion_matrix_metric = BinaryConfusionMatrix(normalize='true')

    confusion_matrix = confusion_matrix_metric(predicted_as_clean, gt_clean)
    print(confusion_matrix)

    # confusion_matrix = metrics['confusion']
    # print(confusion_matrix)

def with_prism(training_features: torch.Tensor, training_labels: torch.Tensor, noisy_training_labels: torch.Tensor,
                            testing_features: torch.Tensor, testing_labels: torch.Tensor,
                            noisy_testing_labels: torch.Tensor):

    xbm_object = CrossBatchMemory(ContrastiveLoss(), embedding_size=2, memory_size=20000, miner=None)

    miner = PRISM(num_classes=10, cross_batch_memory_object=xbm_object)

    xbm_object(training_features, noisy_training_labels)

    miner._last_clean_labels = torch.unique(noisy_training_labels)
    miner._update_centroids()

    miner(testing_features.cuda(), noisy_testing_labels.cuda())

    predicted_as_clean = miner._clean_in_batch
    gt_clean = noisy_testing_labels == testing_labels

    confusion_matrix_metric = BinaryConfusionMatrix(normalize='true')
    confusion_matrix = confusion_matrix_metric(predicted_as_clean, gt_clean)
    print(confusion_matrix)


if __name__ == '__main__':
    seed_everything(13)
    training_features, training_labels = load_data(train=True)
    testing_features, testing_labels = load_data(train=False)
    noisy_training_labels, _ = disturb_targets_symmetric(training_labels, perc=0.5)
    noisy_testing_labels, _ = disturb_targets_symmetric(testing_labels, perc=0.5)

    # Good for classification
    # approach, algorithm_parameters = \
    #     ('EVM', "--distance_metric cosine --distance_multiplier 0.7 --tailsize 0.20 --cover_threshold 0.7")

    for ts in torch.arange(0.1, 1.1, 0.1):
        ts = round(float(ts), 1)
        print(f'Examining tail size: {ts}')

        approach, algorithm_parameters = \
            ('EVM', f'--distance_metric euclidean --distance_multiplier 0.7 --tailsize {ts} --cover_threshold 0.7')
        # approach, algorithm_parameters = \
        #     ('OpenMax', "--distance_metric cosine --distance_multiplier 0.7 --tailsize 0.20")

        raw_openset(approach, algorithm_parameters, training_features, training_labels, noisy_training_labels,
                    testing_features, testing_labels, noisy_testing_labels)

    # random_indices = torch.randperm(len(noisy_testing_labels))[:len(noisy_testing_labels)]
    #
    # with_rejection_strategy(approach, algorithm_parameters, training_features, training_labels, noisy_training_labels,
    #             testing_features[random_indices], testing_labels[random_indices], noisy_testing_labels[random_indices])

    # with_prism(training_features, training_labels, noisy_training_labels, testing_features[random_indices],
    #            testing_labels[random_indices], noisy_testing_labels[random_indices])
