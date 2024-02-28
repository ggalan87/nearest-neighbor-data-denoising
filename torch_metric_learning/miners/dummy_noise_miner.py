import sys
from functools import partial
from typing import Type
import torch
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.miners import BaseMiner

from .common import filter_indices_tuple, get_all_pairs_indices


class DummyNoiseRejection:
    def __init__(self):
        self._current_batch_noisy_samples = None

    @property
    def current_batch_noisy_samples(self):
        return self._current_batch_noisy_samples

    @current_batch_noisy_samples.setter
    def current_batch_noisy_samples(self, value):
        self._current_batch_noisy_samples = value


def post_filtering_check(current_batch_noisy_samples, indices_tuple):
    a1_idx, p_idx, a2_idx, n_idx = indices_tuple

    # Few post checks just to be sure that the correct indices were excluded
    batch_size = len(current_batch_noisy_samples)
    n_noisy = 0
    for i in range(batch_size):
        # Check if the sample is noisy

        if torch.eq(current_batch_noisy_samples[i], True):
            n_noisy += 1

            assert i not in a1_idx and \
                   i not in p_idx and \
                   i not in a2_idx and \
                   i not in n_idx

    assert torch.count_nonzero(current_batch_noisy_samples) == n_noisy


def get_dummy_denoised_pairs_indices(labels, ref_labels=None, noise_aware_object: DummyNoiseRejection = None):
    # Initially, call the function which returns all the indices
    a1_idx, p_idx, a2_idx, n_idx = get_all_pairs_indices(labels, ref_labels)

    # If noise aware approach is not preferred I simply return all the indices
    if noise_aware_object is None:
        return a1_idx, p_idx, a2_idx, n_idx

    a1_idx, p_idx, a2_idx, n_idx = \
        filter_indices_tuple(indices_tuple=get_all_pairs_indices(labels, ref_labels),
                             batch_noise_predictions=noise_aware_object.current_batch_noisy_samples)

    post_filtering_check(current_batch_noisy_samples=noise_aware_object.current_batch_noisy_samples,
                         indices_tuple=(a1_idx, p_idx, a2_idx, n_idx))

    return a1_idx, p_idx, a2_idx, n_idx


def create_dummy_miner(base: Type[BaseMiner], **instance_arguments):
    class DummyNoiseMiner(base):
        def __init__(self, distance_kwargs=None, **kwargs):
            super().__init__(**kwargs)

            self.dummy_noise_rejection = DummyNoiseRejection()

            sys.modules[
                'pytorch_metric_learning'].utils.loss_and_miner_utils.get_all_pairs_indices = \
                partial(get_dummy_denoised_pairs_indices, noise_aware_object=self.dummy_noise_rejection)

        def forward(self, embeddings, labels, ref_emb=None, ref_labels=None, noisy_samples=None):
            self.dummy_noise_rejection.current_batch_noisy_samples = noisy_samples
            return super().forward(embeddings, labels, ref_emb, ref_labels)

    return DummyNoiseMiner(**instance_arguments)


