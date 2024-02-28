import torch
from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import HighScoreInPositiveClassCriterion, \
    InclusionInPositiveClassCriterion, LowScoreInNegativeClassCriterion, NonInclusionInNegativeClassCriterion, \
    HighestIsSoloCriterion, OpenSetClassifierIsCorrect, KLDivergenceIsLowCriterion, CombinedCriteria
from torch_metric_learning.noise_reducers.sample_rejection import *


class TestRejectionCriteria:
    def test_positive_is_clean(self):
        criterion = InclusionInPositiveClassCriterion()
        label_idx = 0
        # prediction is 1, therefore included -> clean
        predictions = torch.tensor([1, 1, 1, 0, 0])
        assert criterion.sample_is_clean(label_idx, predictions)

    def test_positive_is_noisy(self):
        criterion = InclusionInPositiveClassCriterion()
        label_idx = 0
        # prediction is 0, therefore included -> clean
        predictions = torch.tensor([0, 1, 1, 0, 0])
        assert criterion.sample_is_noisy(label_idx, predictions)

    def test_positive_is_clean_raw(self):
        criterion = HighScoreInPositiveClassCriterion(threshold=0.5)
        label_idx = 0
        # prediction is 0.8 (larger than threshold), therefore included -> clean
        predictions = torch.tensor([0.8, 0.3, 0.3, 0.9, 0.05])
        assert criterion.sample_is_clean(label_idx, predictions)

    def test_positive_is_noisy_raw(self):
        criterion = HighScoreInPositiveClassCriterion(threshold=0.5)
        label_idx = 0
        # prediction is 0.4 (lower than threshold), therefore included -> clean
        predictions = torch.tensor([0.4, 0.3, 0.3, 0.9, 0.05])
        assert criterion.sample_is_noisy(label_idx, predictions)

    def test_negative_is_clean(self):
        criterion = NonInclusionInNegativeClassCriterion()
        label_idx = 0
        # prediction is 1 for the label and zero for the rest, therefore not included in other classes -> clean
        predictions = torch.tensor([1, 0, 0, 0, 0])
        assert criterion.sample_is_clean(label_idx, predictions)

    def test_negative_is_noisy(self):
        criterion = NonInclusionInNegativeClassCriterion()
        label_idx = 0
        # prediction is 1 for the label and 1 for at least one other class, therefore included in another class -> noisy
        predictions = torch.tensor([1, 1, 0, 0, 0])
        assert criterion.sample_is_noisy(label_idx, predictions)

    def test_negative_is_clean_raw(self):
        criterion = LowScoreInNegativeClassCriterion(threshold=0.5)
        label_idx = 0
        # prediction for classes other than 0 are lower than threshold, therefore not included -> clean
        predictions = torch.tensor([0.8, 0.3, 0.3, 0.2, 0.05])
        assert criterion.sample_is_clean(label_idx, predictions)

    def test_negative_is_noisy_raw(self):
        criterion = LowScoreInNegativeClassCriterion(threshold=0.5)
        label_idx = 0
        # prediction for a class other than 0 is higher than threshold, therefore included in another class -> noisy
        predictions = torch.tensor([0.8, 0.3, 0.3, 0.9, 0.05])
        assert criterion.sample_is_noisy(label_idx, predictions)

    ################
    def test_positives_are_clean(self):
        criterion = InclusionInPositiveClassCriterion()
        label_inidices = torch.tensor([0, 0])
        # prediction is 1, therefore included -> clean
        predictions = \
            torch.tensor([
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0]
            ])

        samples_are_clean = criterion.sample_is_clean(label_inidices, predictions)
        assert torch.all(samples_are_clean)

    def test_positives_are_noisy(self):
        criterion = InclusionInPositiveClassCriterion()
        label_inidices = torch.tensor([0, 0])
        # prediction is 1, therefore included -> clean
        predictions = \
            torch.tensor([
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0]
            ])

        samples_are_noisy = criterion.sample_is_noisy(label_inidices, predictions)
        assert torch.all(samples_are_noisy)

    def test_negatives_are_noisy(self):
        criterion = NonInclusionInNegativeClassCriterion()
        label_inidices = torch.tensor([0, 0])
        predictions = \
            torch.tensor([
                [1, 0, 1, 0, 0],
                [1, 0, 1, 0, 0]
            ])

        samples_are_noisy = criterion.sample_is_noisy(label_inidices, predictions)
        assert torch.all(samples_are_noisy)

    def test_negatives_are_noisy_raw(self):
        criterion = LowScoreInNegativeClassCriterion(threshold=0.5)
        label_inidices = torch.tensor([0, 2])
        predictions = \
            torch.tensor([
                [0.8, 0.2, 0.6, 0.1, 0.35],
                [0.6, 0.4, 0.9, 0.05, 0.06]
            ])

        samples_are_noisy = criterion.sample_is_noisy(label_inidices, predictions)
        assert torch.all(samples_are_noisy)

    def test_highest_is_solo_clean(self):
        criterion = HighestIsSoloCriterion(threshold=0.1)
        label_inidices = torch.tensor([0, 2])
        predictions = \
            torch.tensor([
                [0.4, 0.2, 0.1, 0.20, 0.1],
                [0.1, 0.2, 0.4, 0.20, 0.1],
            ])

        samples_are_clean = criterion.sample_is_clean(label_inidices, predictions)
        assert torch.all(samples_are_clean)

    def test_highest_is_solo_noisy(self):
        criterion = HighestIsSoloCriterion(threshold=0.1)
        label_inidices = torch.tensor([0, 2])
        predictions = \
            torch.tensor([
                [0.4, 0.39, 0.01, 0.0, 0.0],
                [0.2, 0.30, 0.35, 0.15, 0.0]
            ])

        samples_are_noisy = criterion.sample_is_noisy(label_inidices, predictions)
        assert torch.all(samples_are_noisy)

    def test_openset_classifier_is_correct_clean(self):
        criterion = OpenSetClassifierIsCorrect(threshold=0.5)
        label_inidices = torch.tensor([0, 2])
        predictions = \
            torch.tensor([
                [0.8, 0.2, 0.1, 0.20, 0.1],
                [0.1, 0.2, 0.8, 0.20, 0.1],
            ])

        samples_are_clean = criterion.sample_is_clean(label_inidices, predictions)
        assert torch.all(samples_are_clean)

    def test_openset_classifier_is_correct_noisy(self):
        criterion = OpenSetClassifierIsCorrect(threshold=0.5)
        label_inidices = torch.tensor([0, 2])
        predictions = \
            torch.tensor([
                [0.4, 0.39, 0.01, 0.0, 0.0],
                [0.2, 0.7, 0.35, 0.15, 0.0]
            ])

        samples_are_noisy = criterion.sample_is_noisy(label_inidices, predictions)
        assert torch.all(samples_are_noisy)

    def test_kl_divergence_is_low_clean(self):
        criterion = KLDivergenceIsLowCriterion(threshold=0.99)
        label_inidices = torch.tensor([0, 0, 2])
        predictions = \
            torch.tensor([
                [0.9, 0.0, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.1, 0.20, 0.1],
                [0.1, 0.2, 0.8, 0.20, 0.1],
            ])

        samples_are_clean = criterion.sample_is_clean(label_inidices, predictions)
        assert torch.all(samples_are_clean)

    def test_kl_divergence_is_low_noisy(self):
        criterion = KLDivergenceIsLowCriterion(threshold=0.99)
        label_inidices = torch.tensor([0, 2])
        predictions = \
            torch.tensor([
                [0.4, 0.39, 0.01, 0.0, 0.0],
                [0.2, 0.7, 0.35, 0.15, 0.0]
            ])

        samples_are_noisy = criterion.sample_is_noisy(label_inidices, predictions)
        assert torch.all(samples_are_noisy)

    def test_multiple_criteria(self):
        label_indices = \
            torch.tensor([
                0, 1, 2, 3, 4
            ])

        predictions = \
            torch.tensor([
                # 0    1    2    3    4
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.6, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.85, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.45, 0.8],
            ])

        gt_noisy = torch.tensor([False, True, True, True, False])

        criteria = [
            # False, False, True, False, False
            HighScoreInPositiveClassCriterion(threshold=0.5),
            # False, True, True, True, False
            LowScoreInNegativeClassCriterion(threshold=0.5),
            # False, False, False, True, False
            HighestIsSoloCriterion(threshold=0.1)
        ]
        combined = CombinedCriteria(criteria)

        # Test either ways
        assert torch.equal(combined.sample_is_noisy(label_indices, predictions), gt_noisy) and \
            torch.equal(combined.sample_is_clean(label_indices, predictions), torch.logical_not(gt_noisy))
