from typing import Literal

import torch
from pytorch_metric_learning.utils.inference import FaissKNN
from evt.vast_openset import OpensetTrainer


def calculate_density_weights(labels, n_labels, indices, k_neighbors, scale=True):
    # TODO: Optimize the loop
    # Computes the histogram of precomputed values
    try:
        counts = torch.stack(
            [torch.bincount(labels[indices[i][:k_neighbors]], minlength=n_labels) for i in
             range(len(indices))])
    except RuntimeError as e:
        raise e

    weights = counts / k_neighbors  # len(self._data.labels_to_indices)

    # Scale weights into essentially 0-1 range from 0-small_quantity
    # TODO: Consider result.max(dim=1).values instead of 1.0

    if scale:
        weights *= (1.0 / weights.max(dim=1).values).expand(weights.shape[1], weights.shape[0]).T

    return weights


def calculate_variable_density_weights(labels, n_labels, indices, k_neighbors,
                                       reduce_type: Literal['none', 'median'] = 'median'):
    weights_list = []

    k = k_neighbors
    stop_k = max(int(k_neighbors / 20), 5)
    while k >= stop_k:
        weights_list.append(calculate_density_weights(labels, n_labels, indices, k))
        k = int(k / 2)

    if reduce_type == 'none':
        return torch.stack(weights_list)
    elif reduce_type == 'median':
        return torch.median(torch.stack(weights_list), dim=0).values
    else:
        raise NotImplementedError


def knn_density(features, labels, n_labels, knn_func, k_neighbors, magnitude):
    # Find knn indices
    distances, indices = knn_func(features, k_neighbors)
    indices = indices.cpu()
    weights = calculate_density_weights(labels, n_labels, indices, k_neighbors)
    # weights = calculate_variable_density_weights(labels, n_labels, indices, k_neighbors)

    # Zero magnitude is another way to mute the effect of weighting
    return magnitude * weights if magnitude > 0.0 else 1.0


def variable_good_indices(indices, label, training_labels, n_labels):
    """ This function applies variable k for KNN and predicts as good if the """

    # A vector full of zeros
    good_indices = torch.zeros((indices.shape[0], ), dtype=torch.bool)

    k = indices.shape[1]

    variable_weights = calculate_variable_density_weights(training_labels, n_labels, indices, k, reduce_type='none')

    for w in variable_weights:
        # The vector is updated with goodness from the variable weights (computed for variable k)
        # The sample is good if the indices correspond to the correct class label
        good_indices = good_indices | torch.argmax(w, dim=1) == int(label)

    return good_indices


class KNNTrainer:
    def __init__(self, k_neighbors, magnitude, use_for_openset):
        self._k_neighbors = k_neighbors
        self._magnitude = magnitude
        self._use_for_openset = use_for_openset
        self._knn_func = None

        self._openset_trainer = None

    def _train(self, trainer: OpensetTrainer):
        # Keep as needed for prediction
        self._openset_trainer = trainer

        training_feats, training_labels = trainer.data.back_convert_data()

        self._knn_func = FaissKNN(reset_before=False, reset_after=False)
        self._knn_func.train(training_feats)

        if self._use_for_openset:
            k_neighbors = self._k_neighbors

            modified_features_dict = {}
            for label, features in trainer.data.features_dict.items():
                distances, indices = self._knn_func(features, k_neighbors)

                weights = calculate_density_weights(training_labels, len(trainer.data.labels_to_indices), indices,
                                                    k_neighbors)

                # Keep only those samples which correspond to "good", i.e. most neighbors come from the true class
                good_indices = torch.argmax(weights, dim=1) == int(label)

                # TODO: introduce option for variable K
                # good_indices = variable_good_indices(indices, label, training_labels, len(trainer.data.labels_to_indices))

                modified_features_dict[label] = features[good_indices]

            # Keep only the reduced set of features per class
            for label, features in modified_features_dict.items():
                trainer.data.features_dict[label] = features

            # Retrain the knn using kept data
            training_feats, _ = self._openset_trainer.data.back_convert_data()
            self._knn_func.train(training_feats)

    def __call__(self, trainer: OpensetTrainer):
        self._train(trainer)

    def predict(self, features):
        openset_data = self._openset_trainer.data
        _, training_labels = openset_data.back_convert_data()

        return knn_density(features=features, labels=training_labels, n_labels=len(openset_data.labels_to_indices),
                           knn_func=self._knn_func, k_neighbors=self._k_neighbors, magnitude=self._magnitude)
