from abc import abstractmethod
from typing import Dict, List, Union

import torch
from pytorch_metric_learning.distances.lp_distance import LpDistance
from evt.vast_openset import OpensetTrainer
from lightning.ext import logger


class SelectionStrategy:
    """
    Base class for defining strategies about which weibulls to select for keeping in the model
    """
    def __init__(self):
        self._distance = LpDistance()

    def process_model(self, class_features, class_model) -> torch.Tensor:
        n_weibs = len(class_model['extreme_vectors'])
        n_class_samples = len(class_features)

        # TODO: document logic

        weibs_scores = []

        # In this loop we score all weibulls
        for i in range(n_weibs):
            # Find the indices which correspond to the samples covered by the i-th weibull
            weib_samples_indices = class_model['covered_vectors'][i]

            # Ignore weibulls which are supported by a single sample
            if weib_samples_indices.shape == torch.Size([]):
                weibs_scores.append(0)
                continue

            # Obtain the relevant features
            weib_samples_features = class_features[weib_samples_indices]
            # Compute and store the score
            weibs_scores.append(self._get_score(weib_samples_features))

        # Convert to tensor
        weibs_scores = torch.tensor(weibs_scores)
        covered_vectors = class_model['covered_vectors']
        return self._strategy_impl(weibs_scores, n_class_samples, covered_vectors)

    @abstractmethod
    def _strategy_impl(self, weibs_scores, n_class_samples, covered_vectors) -> torch.Tensor:
        pass

    def _get_score(self, weib_samples_features):
        n_samples = len(weib_samples_features)
        # compute pairwise distances
        dist_mat = self._distance(weib_samples_features)
        # compute variance of these distances
        dist_var = torch.std(torch.triu(dist_mat, diagonal=1))
        # compute samples over the variance of their pairwise distances ratio
        return n_samples / dist_var


class SamplesCoverageStrategy(SelectionStrategy):
    """
    This strategy keeps weibull models which are supported by a number of weibull models. The models are first sorted
    according to their score and then models are progressively kept until coverage ratio is surpassed
    """
    def __init__(self, coverage_ratio):
        super().__init__()
        self._coverage_ratio = coverage_ratio

    def _strategy_impl(self, weibs_scores, n_class_samples, covered_vectors) -> torch.Tensor:
        n_weibs = len(weibs_scores)

        # Get indices of high->low scores
        sorted_indices = torch.argsort(weibs_scores, descending=True)

        # Initialize the return vector with all zeros
        good_class_weibulls = torch.zeros(n_weibs, dtype=torch.bool)

        # Based on distance
        n_described = int(self._coverage_ratio * n_class_samples)

        # Each sample may support more than one weibull model, therefore we create a set such that, we keep the unique
        # samples that are covered
        covered_samples = set()
        for i, idx in enumerate(sorted_indices):
            covered_samples.update(covered_vectors[idx].tolist())
            good_class_weibulls[idx] = True
            if len(covered_samples) > n_described:
                logger.debug(
                    f'last good density score: {weibs_scores[idx]}. bad density score: {weibs_scores[sorted_indices[i + 1]]}')
                break

        return good_class_weibulls


class MedianScoreStrategy(SelectionStrategy):
    """
    This strategy keeps weibull models whose score is larger than the median score of all models
    """
    def __init__(self):
        super().__init__()

    def _strategy_impl(self, weibs_scores, n_class_samples, covered_vectors) -> torch.Tensor:
        median_score = torch.median(weibs_scores)
        good_class_weibulls = weibs_scores > median_score
        return good_class_weibulls


class ConstantThresholdStrategy(SelectionStrategy):
    """
    This strategy keeps weibull models whose score is larger than the given threshold
    """
    def __init__(self, threshold):
        super().__init__()

        threshold_is_in_range = 0.0 <= threshold <= 1.0
        if not threshold_is_in_range:
            raise ValueError(f'Threshold must be in [0-1] because it is relative to min,max values per model. '
                             f'{threshold} was given instead')
        self._threshold = threshold

    @staticmethod
    def _minmax_scale(data):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

    def _strategy_impl(self, weibs_scores, n_class_samples, covered_vectors) -> torch.Tensor:
        good_class_weibulls = self._minmax_scale(weibs_scores) > self._threshold
        return good_class_weibulls


class DensityBasedModelReduction:
    """
    Reduces the computed openset models according to some density criterion.
    """
    def __init__(self, strategy: Union[str, List], strategy_options: Dict):
        strategy = strategy if isinstance(strategy, list) else [strategy]
        strategy_options = strategy_options

        self._strategies_list = []

        mapping = \
            {
                'coverage': SamplesCoverageStrategy(strategy_options.get('samples_coverage_ratio')),
                'median': MedianScoreStrategy(),
                'threshold': ConstantThresholdStrategy(strategy_options.get('score_threshold'))
            }

        for s in strategy:
            try:
                self._strategies_list.append(mapping[s])
            except KeyError:
                raise NotImplementedError(f'Invalid strategy {s}')

    def _reduce_model(self, trainer: OpensetTrainer):
        strategies_list = self._strategies_list

        # Filter weibulls
        for label, model in trainer.models.items():
            class_features = trainer.data.features_dict[label]
            results = [strategy.process_model(class_features, model) for strategy in strategies_list]
            good_class_weibulls = torch.all(torch.stack(results), dim=0)

            model['extreme_vectors'] = model['extreme_vectors'][good_class_weibulls]
            model['extreme_vectors_indexes'] = model['extreme_vectors_indexes'][good_class_weibulls]
            model['weibulls'].wbFits = model['weibulls'].wbFits[good_class_weibulls]
            model['weibulls'].smallScoreTensor = model['weibulls'].smallScoreTensor[good_class_weibulls]

            indices_to_keep = torch.where(good_class_weibulls)[0].tolist()
            model['covered_vectors'] = [i for j, i in enumerate(model['covered_vectors'])
                                        if j in indices_to_keep]

            logger.debug(
                f'Class {label}, kept {torch.count_nonzero(good_class_weibulls)} out of {len(good_class_weibulls)} '
                f'from #{len(class_features)} samples')

    def __call__(self, trainer: OpensetTrainer):
        self._reduce_model(trainer)
