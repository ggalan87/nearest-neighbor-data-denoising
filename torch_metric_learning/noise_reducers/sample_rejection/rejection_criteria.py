from abc import abstractmethod
from typing import Literal

import torch
import torch.nn.functional as F
from torchmetrics.regression import KLDivergence


class RejectionCriterion:
    def sample_is_noisy(self, label_idx, predictions) -> torch.Tensor:
        label_idx, predictions = self._check_and_fix_input(label_idx, predictions)
        return self._sample_is_noisy_impl(label_idx, predictions)

    def sample_is_clean(self, label_idx, predictions) -> torch.Tensor:
        return torch.logical_not(self.sample_is_noisy(label_idx, predictions))

    @abstractmethod
    def _sample_is_noisy_impl(self, label_idx, predictions) -> torch.Tensor:
        pass

    @staticmethod
    def _check_and_fix_input(label_idx, predictions):
        if isinstance(label_idx, torch.Tensor) and len(predictions.shape) == 2:
            assert len(label_idx) == len(predictions)
            label_idx = label_idx.reshape(len(predictions), 1).to(device=predictions.device)
        elif isinstance(label_idx, int) and len(predictions.shape) == 1:
            label_idx = torch.tensor([label_idx])
            predictions = predictions.reshape(1, len(predictions))
            assert len(label_idx) == len(predictions)
            label_idx = label_idx.reshape(len(predictions), 1).to(device=predictions.device)
        else:
            assert False

        return label_idx, predictions

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v!r}' for k, v in self.__dict__.items() if not k.startswith('__')])})"


class HighScoreInPositiveClassCriterion(RejectionCriterion):
    def __init__(self, threshold):
        self._threshold = threshold

    def _sample_is_noisy_impl(self, label_idx, predictions) -> torch.Tensor:
        # The sample is rejected as noisy if the class probability corresponding to its label is lower than threshold
        return (predictions.gather(dim=1, index=label_idx) < self._threshold).ravel()


class InclusionInPositiveClassCriterion(RejectionCriterion):
    def __init__(self):
        pass

    def _sample_is_noisy_impl(self, label_idx, predictions) -> torch.BoolTensor:
        # The sample is rejected as noisy if the class probability corresponding to its label is lower than threshold
        return (predictions.gather(dim=1, index=label_idx) != 1).ravel()


class LowScoreInNegativeClassCriterion(RejectionCriterion):
    def __init__(self, threshold):
        self._threshold = threshold

    def _sample_is_noisy_impl(self, label_idx, predictions: torch.Tensor) -> torch.Tensor:
        # We need a mask filled with True in indices which correspond to other classes. We first create a mask
        # initialized by all True and then assign False to same class indices. This is done per row, using the
        # scatter_ method
        selected_indices = torch.ones(predictions.shape, dtype=torch.bool)
        selected_indices.scatter_(dim=1, index=label_idx, src=torch.zeros_like(label_idx, dtype=torch.bool))

        # Afterwards we seek for those indices which have probabilities larger that the specified threshold and are
        # among the selected indices (i.e. corresponding to other classes). To do so, we apply the bitwise and operation
        # between the indices which surpass the threshold and the (per row) non-class indices
        bad_indices = torch.bitwise_and(predictions >= self._threshold, selected_indices)

        # Report True if any of the bad_indices per row are True
        # From the docs: For each row of input in the given dimension dim, returns True if any element in the row
        # evaluate to True and False otherwise.
        return torch.any(bad_indices, dim=1)


class NonInclusionInNegativeClassCriterion(RejectionCriterion):
    def __init__(self):
        pass

    def _sample_is_noisy_impl(self, label_idx, predictions) -> torch.Tensor:
        # Method same as above, except criterion explanation omitted
        selected_indices = torch.ones(predictions.shape, dtype=torch.bool)
        selected_indices.scatter_(dim=1, index=label_idx, src=torch.zeros_like(label_idx, dtype=torch.bool))
        bad_indices = torch.bitwise_and(predictions == 1, selected_indices)
        return torch.any(bad_indices, dim=1)


class HighestIsSoloCriterion(RejectionCriterion):
    def __init__(self, threshold):
        self._threshold = threshold

    @staticmethod
    def _check_and_fix_input(label_idx, predictions):
        label_idx, predictions = RejectionCriterion._check_and_fix_input(label_idx, predictions)

        # Ensure that each row sums to 1.0. Chose to not throw exception
        predictions = torch.nn.functional.normalize(predictions, p=1.0)

        return label_idx, predictions

    def _sample_is_noisy_impl(self, label_idx, predictions) -> torch.Tensor:
        top_two_per_sample = predictions.topk(2).values
        return top_two_per_sample[:, 0] < top_two_per_sample[:, 1] + self._threshold


class OpenSetClassifierIsCorrect(RejectionCriterion):
    def __init__(self, threshold):
        self._threshold = threshold

    def _sample_is_noisy_impl(self, label_idx, predictions) -> torch.Tensor:
        max_prediction_index = torch.argmax(predictions, dim=1)

        wrong_label_is_predicted = max_prediction_index != label_idx.ravel()
        predicted_label_has_low_probability = \
            (predictions.gather(dim=1, index=max_prediction_index.reshape(label_idx.shape)) < self._threshold).ravel()
        return torch.logical_or(wrong_label_is_predicted, predicted_label_has_low_probability)


class KLDivergenceIsLowCriterion(RejectionCriterion):
    def __init__(self, threshold,):
        self._threshold = threshold

    def _sample_is_noisy_impl(self, label_idx, predictions) -> torch.Tensor:
        # reduction is set to none because we want to keep per sample divergence
        kl_divergence = KLDivergence(reduction='none')
        true_label_distributions = torch.zeros_like(predictions) + torch.finfo(torch.float32).eps
        selected_indices = torch.zeros(true_label_distributions.shape, dtype=torch.bool)
        selected_indices.scatter_(dim=1, index=label_idx, src=torch.ones_like(label_idx, dtype=torch.bool))
        true_label_distributions[selected_indices] = 1.0

        # Both distributions should express probabilities, therefore we normalize across dim=1 (default)
        predictions = F.normalize(predictions, p=1.0)
        true_label_distributions = F.normalize(true_label_distributions, p=1.0)

        divergences = 1 - torch.exp(-kl_divergence(predictions, true_label_distributions))
        return divergences > self._threshold


class CombinedCriteria:
    def __init__(self, criteria_list, reduction_type: Literal['any', 'max'] = 'any'):
        self._criteria_list = criteria_list
        self._reduction_type = reduction_type

        if reduction_type not in ['any', 'max']:
            raise ValueError('Unsupported reduction type!')

    def sample_is_noisy(self, label_idx, predictions):
        results_noisy = torch.stack([criterion.sample_is_noisy(label_idx, predictions)
                                     for criterion in self._criteria_list])

        if self._reduction_type == 'any':
            return torch.any(results_noisy, dim=0)
        else:
            # If > 50% of the criteria say noisy then it is noisy. Consider fallback to any in case of 2 criteria
            return (torch.count_nonzero(results_noisy, dim=0) / len(self._criteria_list)) > 0.5

    def sample_is_clean(self, label_idx, predictions):
        return torch.logical_not(self.sample_is_noisy(label_idx, predictions))
