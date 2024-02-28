import pickle
from pathlib import Path
from typing import Union, Dict

import torch
from torch.nn import functional as F

from torch_metric_learning.noise_reducers.memory_bank import MemoryBank
from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import RejectionCriterion, \
    CombinedCriteria


class RejectionStrategy:
    # TODO: Decouple memory bank from rejection strategy
    def __init__(self, training_samples_fraction, rejection_criteria: Union[RejectionCriterion, CombinedCriteria],
                 noisy_positive_as_negative=False, use_raw_probabilities=False):
        self._training_samples_fraction = training_samples_fraction
        self._rejection_criteria = rejection_criteria
        self._noisy_positive_as_negative = noisy_positive_as_negative
        self._use_raw_probabilities = use_raw_probabilities

        if use_raw_probabilities:
            # Added this until I correctly reimplemented using the new APIs
            raise NotImplementedError

        # Stores batch predictions, a boolean tensor with shape (batch_size,). True: noisy, False: clean
        self._batch_noise_predictions = None

        # GT noisy - this is for evaluating the rejection method against the ground-truth values
        self._batch_noise_gt = None

        # Stores batch raw scores, a float tensor of shape (batch_size, #classes)
        self._batch_raw_scores = None

        # Used only if raw probabilities are enabled, for storing the computed weights
        self._batch_weights = torch.tensor([1]).to('cuda:0')

    def _compute_noise_predictions(self, labels):
        batch_size = len(labels)

        # Initialize with all False (clean), and then pass the samples to see which are noisy
        self._batch_noise_predictions = torch.zeros((batch_size,), dtype=torch.bool)

        for i in range(batch_size):
            sample_scores = self._batch_raw_scores[i]
            label_idx = self.labels_to_indices[int(labels[i])]

            if self._rejection_criteria.sample_is_noisy(label_idx, sample_scores):
                self._batch_noise_predictions[i] = True

        # print(f'Found {torch.count_nonzero(self._batch_noise_predictions)} noisy samples.')

    def train(self, memory_bank: MemoryBank):
        """
        Normally the memory_bank needs to be shown only once to the object, because it is the same object,
        but I leave as such in order to support possible dynamic setups or the case that some implementations do not
        keep the object as member
        """
        raise NotImplementedError

    def has_trained(self):
        raise NotImplementedError

    def _store_batch_raw_scores(self, embeddings, labels, normalize=True):
        raise NotImplementedError

    def predict_noise(self, embeddings, labels, normalize=True):
        self._store_batch_raw_scores(embeddings, labels, normalize)
        self._compute_noise_predictions(labels)

        return self.current_batch_noisy_predictions

    def retrieve_batch_noise_predictions(self):
        return self._batch_noise_predictions

    def retrieve_batch_raw_scores(self):
        return self._batch_raw_scores

    def store_batch_weights(self, batch_weights: torch.Tensor):
        if not self.use_raw_probabilities:
            raise AssertionError('Invalid configuration. Setting batch weights while not using raw probabilities')
        self._batch_weights = batch_weights

    def retrieve_batch_weights(self) -> torch.Tensor:
        if not self.use_raw_probabilities:
            raise AssertionError('Invalid configuration. Retrieving batch weights while not using raw probabilities')
        return self._batch_weights

    @property
    def labels_to_indices(self) -> Dict[int, int]:
        raise NotImplementedError

    @property
    def noisy_positive_as_negative(self):
        return self._noisy_positive_as_negative

    @property
    def use_raw_probabilities(self):
        return self._use_raw_probabilities

    @property
    def current_batch_noisy_samples(self):
        return self._batch_noise_gt

    @current_batch_noisy_samples.setter
    def current_batch_noisy_samples(self, value):
        self._batch_noise_gt = value

    @property
    def current_batch_noisy_predictions(self):
        return self._batch_noise_predictions


class Inspector:
    def __init__(self, output_dir: str = './rejection_inspector_output'):
        self._predictions_scores_list = []
        self._predictions_noisy_list = []
        self._labels_list = []
        self._noisy_indices_list = []

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True)

        self._epoch = 0

    def add_batch_info(self, batch_labels: torch.Tensor, batch_predictions_scores: torch.Tensor,
                       batch_predictions_noisy: torch.Tensor, batch_gt_noisy: torch.Tensor):
        self._predictions_scores_list.append(batch_predictions_scores)
        self._predictions_noisy_list.append(batch_predictions_noisy)
        self._labels_list.append(batch_labels)
        self._noisy_indices_list.append(batch_gt_noisy)

    def report_and_reset(self, labels_to_indices):
        all_labels = torch.hstack(self._labels_list)

        pickle.dump(labels_to_indices, open(self._output_dir / f'labels_to_indices-{self._epoch}.pkl', 'wb'))
        torch.save(all_labels, self._output_dir / f'all_labels-{self._epoch}.pt')

        # Noisy indices sometimes are not available
        if len(self._noisy_indices_list) > 0 and self._noisy_indices_list[0] is not None:
            all_noisy_indices = torch.hstack(self._noisy_indices_list)
            torch.save(all_noisy_indices, self._output_dir / f'all_noisy_indices-{self._epoch}.pt')

        # If rejection strategy has not yet trained, then these values are None / irrelevant
        if self._predictions_scores_list[0] is not None:
            all_predictions_scores = torch.vstack(self._predictions_scores_list)
            torch.save(all_predictions_scores, self._output_dir / f'all_predictions_scores-{self._epoch}.pt')

        if self._predictions_noisy_list[0] is not None:
            all_predictions_noisy = torch.hstack(self._predictions_noisy_list)
            torch.save(all_predictions_noisy, self._output_dir / f'all_predictions_noisy-{self._epoch}.pt')

        self._predictions_scores_list = []
        self._predictions_noisy_list = []
        self._labels_list = []
        self._noisy_indices_list = []

        self._epoch += 1
