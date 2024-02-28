from dataclasses import asdict
from typing import Union, Optional, Dict

import torch
from torch.nn import functional as F

from evt.vast_openset import OpensetModelParameters, OpensetData, OpensetTrainer
from torch_metric_learning.noise_reducers.memory_bank import MemoryBank
from torch_metric_learning.noise_reducers.sample_rejection.openset_helpers import NoiseHandlingOptions
from lightning.losses_playground import ExclusionInspector
from torch_metric_learning.noise_reducers.sample_rejection.openset_helpers.density_based_model_reduction import \
    DensityBasedModelReduction
from torch_metric_learning.noise_reducers.sample_rejection.openset_helpers.knn_based_reweighting import KNNTrainer
from torch_metric_learning.noise_reducers.sample_rejection.rejection_base import RejectionStrategy
from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import RejectionCriterion, \
    CombinedCriteria, HighScoreInPositiveClassCriterion


class PopulationAwarePairRejection(RejectionStrategy):
    def __init__(self, approach='OpenMax',
                 algorithm_parameters_string='--distance_metric euclidean --tailsize 0.25',
                 training_samples_fraction=0.3,
                 rejection_criteria: Union[RejectionCriterion, CombinedCriteria] = HighScoreInPositiveClassCriterion(
                     threshold=0.5),
                 noisy_positive_as_negative=False,
                 use_raw_probabilities=False,
                 noise_handling_options: Optional[NoiseHandlingOptions] = None):

        super().__init__(training_samples_fraction=training_samples_fraction,
                         rejection_criteria=rejection_criteria,
                         noisy_positive_as_negative=noisy_positive_as_negative,
                         use_raw_probabilities=use_raw_probabilities)

        # TODO: Expose in config.yaml

        # if approach == 'EVM':
        #     algorithm_parameters_string += " --distance_multiplier 0.7"

        saver_parameters = f"--OOD_Algo {approach}"
        self._openset_model_params = OpensetModelParameters(approach, algorithm_parameters_string, saver_parameters)

        # Initially openset trainer is None, because no training has been done. I use it as a flag for bypassing the
        # rejection code path
        self._openset_trainer = None

        self._knn_trainer = None

        # The indices of the input data that were used due to 'training_samples_fraction'
        self._random_global_indices = None

        if noise_handling_options is not None and noise_handling_options.nn_weights is not None \
                and not use_raw_probabilities:
            raise ValueError('KNN weighting is compatible only with raw probabilities')

        self._noise_handling_options = noise_handling_options

        self.ei = ExclusionInspector()

    def train(self, memory_bank: MemoryBank):
        """
        Trains openset model with the data that have been collected so far from the latest epoch
        """

        all_features, all_labels, all_indices, random_indices = memory_bank.get_data(self._training_samples_fraction)

        self._random_global_indices = random_indices

        # Convert the data into appropriate format
        training_feats = OpensetData(features=all_features.to(torch.float32), class_labels=all_labels,
                                     dataset_indices=all_indices)
        # Utilize a new instance of the openset trainer using the latest embeddings
        self._openset_trainer = OpensetTrainer(training_feats, self._openset_model_params, inference_threshold=0.5)

        if self._noise_handling_options.nn_weights is not None:
            self._knn_trainer = KNNTrainer(**asdict(self._noise_handling_options.nn_weights))
            self._openset_trainer.register_hook('before_training', self._knn_trainer)

        if self._noise_handling_options.density_mr is not None:
            model_reduction = DensityBasedModelReduction(**asdict(self._noise_handling_options.density_mr))
            inspection_callback = self._noise_handling_options.inspection_callback

            # Add callback is not None, do inspection once before and once after the model reduction
            if inspection_callback is not None:
                self._openset_trainer.register_hook('after_training', inspection_callback)
            self._openset_trainer.register_hook('after_training', model_reduction)
            if inspection_callback is not None:
                self._openset_trainer.register_hook('after_training', inspection_callback)

        print('Openset training')
        self._openset_trainer.train()

    def has_trained(self):
        return self._openset_trainer is not None

    def _store_batch_raw_scores(self, embeddings, labels, normalize=True):
        # Inclusion probabilities - NxC
        inclusion_probabilities = self._openset_trainer.get_inclusion_probabilities(F.normalize(embeddings))

        # Normalize them such that they sum up to 1 per sample
        batch_raw_scores = F.normalize(inclusion_probabilities, p=1.0) if normalize else inclusion_probabilities

        if self._knn_trainer is None:
            self._batch_raw_scores = batch_raw_scores
        else:
            knn_weights = self._knn_trainer.predict(embeddings)
            self._batch_raw_scores = knn_weights * batch_raw_scores

    @property
    def labels_to_indices(self) -> Dict[int, int]:
        return self._openset_trainer.data.labels_to_indices

    @property
    def random_global_indices(self) -> torch.Tensor:
        return self._random_global_indices
