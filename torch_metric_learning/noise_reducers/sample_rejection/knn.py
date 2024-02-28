import warnings
from typing import Union, Dict

import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import FaissKNN

from torch_metric_learning.noise_reducers.memory_bank import MemoryBank
from torch_metric_learning.noise_reducers.sample_rejection.openset_helpers.knn_based_reweighting import knn_density
from torch_metric_learning.noise_reducers.sample_rejection.rejection_base import RejectionStrategy
from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import RejectionCriterion, \
    CombinedCriteria, HighScoreInPositiveClassCriterion
from torch_metric_learning.utils import BatchedKNN


class KNNPairRejection(RejectionStrategy):
    def __init__(self,
                 num_classes,
                 k_neighbors=20,
                 training_samples_fraction=1.0,
                 rejection_criteria: Union[RejectionCriterion, CombinedCriteria] = HighScoreInPositiveClassCriterion(
                     threshold=0.5),
                 weights_magnitude=1.0,
                 noisy_positive_as_negative=False,
                 use_raw_probabilities=False,
                 use_batched_knn=False
                 ):
        super().__init__(training_samples_fraction=training_samples_fraction,
                         rejection_criteria=rejection_criteria,
                         noisy_positive_as_negative=noisy_positive_as_negative,
                         use_raw_probabilities=use_raw_probabilities)

        self._num_classes = num_classes
        self._k_neighbors = k_neighbors
        self._weights_magnitude = weights_magnitude
        self._knn_func = None
        self._training_labels = None
        self._labels_to_indices = None
        self._use_batched_knn = use_batched_knn

    def train(self, memory_bank: MemoryBank):
        # TODO: Consider single object initialization and multiple call to train function of the same object
        #  but then I have to implement "has_trained" in another way
        self._knn_func = BatchedKNN(LpDistance(normalize_embeddings=True, power=2), batch_size=2048) \
            if self._use_batched_knn else FaissKNN(reset_before=False, reset_after=False)

        all_features, all_labels, all_indices, random_indices = memory_bank.get_data(self._training_samples_fraction)

        self._knn_func.train(all_features.to(torch.float32))
        self._training_labels = all_labels

        # Assume that in the first run all labels are available (initially no noise is found)
        # and therefore can safely construct the dict
        # TODO: This is not true because random identity sampler might throw classes with very few samples
        labels_in_mem = torch.unique(all_labels)

        if self._labels_to_indices is None:
            self._labels_to_indices = {}
            if len(labels_in_mem) == self._num_classes:
                for i, l in enumerate(labels_in_mem):
                    self.labels_to_indices[int(l)] = i
            else:
                warnings.warn(f'Not all labels were in memory {len(labels_in_mem)} vs {self._num_classes}. '
                              f'Will assign same labels to indices, assuming relabeling has occurred and ignoring '
                              f'missing labels.')
                for i, l in enumerate(range(self._num_classes)):
                    self.labels_to_indices[int(l)] = i
    def has_trained(self):
        return self._knn_func is not None

    def _store_batch_raw_scores(self, embeddings, labels, normalize=True):
        training_labels = self._training_labels

        weights = knn_density(features=embeddings, labels=training_labels, n_labels=len(self.labels_to_indices),
                              knn_func=self._knn_func, k_neighbors=self._k_neighbors, magnitude=self._weights_magnitude)

        self._batch_raw_scores = weights

    @property
    def labels_to_indices(self) -> Dict[int, int]:
        return self._labels_to_indices
