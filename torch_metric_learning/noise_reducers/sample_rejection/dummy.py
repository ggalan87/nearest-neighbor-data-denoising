from typing import Dict
from warnings import warn
import torch

from torch_metric_learning.noise_reducers.memory_bank import MemoryBank
from torch_metric_learning.noise_reducers.sample_rejection.rejection_base import RejectionStrategy
from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import CombinedCriteria


class DummyNoiseRejection(RejectionStrategy):
    def __init__(self):
        #
        super().__init__(training_samples_fraction=1.0,
                         # dummy criterion -> combined object with empty list / never called
                         rejection_criteria=CombinedCriteria([]),
                         noisy_positive_as_negative=False,
                         use_raw_probabilities=False)

        self._current_batch_noisy_samples = None
        self._trainer = None

    def train(self, memory_bank: MemoryBank):
        self._trainer = 'dummy'

        # Do explicit reset of the memory
        memory_bank.reset_memory()

    def has_trained(self):
        return self._trainer is not None

    def _store_batch_raw_scores(self, embeddings, labels, normalize=True):
        pass

    def _compute_noise_predictions(self, labels):
        batch_size = len(labels)

        # Initialize with all False (clean), and then pass the samples to see which are noisy
        self._batch_noise_predictions = torch.zeros((batch_size,), dtype=torch.bool)

        # I keep it verbose
        for i in range(batch_size):
            self._batch_noise_predictions[i] = self._current_batch_noisy_samples[i]

    @property
    def labels_to_indices(self) -> Dict[int, int]:
        return {}

    @property
    def current_batch_noisy_samples(self):
        return self._current_batch_noisy_samples

    @current_batch_noisy_samples.setter
    def current_batch_noisy_samples(self, value):
        self._current_batch_noisy_samples = value
