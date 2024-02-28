import sys
from typing import Type, Literal, Tuple
import copy
from functools import partial

from pytorch_metric_learning import miners
from pytorch_metric_learning.miners import BaseMiner
from .passthrough_miner import PassThroughMiner
from .prism import PRISM
from .common import filter_indices_tuple, get_all_pairs_indices

from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import *
from torch_metric_learning.utils import create_distance_patcher
from ..noise_reducers.memory_bank import MemoryBank
from ..noise_reducers.sample_rejection.clustering import ClusterAwarePairRejection
from ..noise_reducers.sample_rejection.knn import KNNPairRejection
from ..noise_reducers.sample_rejection.openset import PopulationAwarePairRejection
from ..noise_reducers.sample_rejection.openset_helpers import NoiseHandlingOptions, NearestNeighborsWeightsOptions, \
    DensityBasedModelsReductionOptions
from ..noise_reducers.sample_rejection.rejection_base import RejectionStrategy, Inspector

IndicesTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def sample_weights_impl(rejection_strategy_object, labels: torch.Tensor, indices_tuple: IndicesTuple):
    # Get raw scores for the batch
    inclusion_probabilities = rejection_strategy_object.retrieve_batch_raw_scores()

    a1_idx, p_idx, a2_idx, n_idx = indices_tuple
    labels_to_indices = rejection_strategy_object.labels_to_indices

    # TODO: Consider implementation with vectorization / smart indexing
    # TODO: Expose ambiguity and reduce function to configuration

    n = len(labels)
    weights_matrix = torch.ones(n, n, device=inclusion_probabilities.device, dtype=torch.float64)

    model_indices_t = torch.tensor(list(map(lambda l: labels_to_indices[int(l)], labels)))

    w_a1 = inclusion_probabilities[a1_idx.cpu(), model_indices_t[a1_idx.cpu()]]
    w_p = inclusion_probabilities[p_idx.cpu(), model_indices_t[p_idx.cpu()]]
    weights_matrix[a1_idx.cpu(), p_idx.cpu()] = w_a1 * w_p

    # for a_i, p_i in zip(a1_idx, p_idx):
    #     # inclusion model for label a. we don't need m_p because by design label of a equals label of p
    #     m_a = model_indices_t[a_i]
    #     w_a = inclusion_probabilities[a_i][m_a]
    #     w_p = inclusion_probabilities[p_i][m_a]
    #     assert weights_matrix[a_i, p_i] == w_a * w_p

    w_a2 = inclusion_probabilities[a2_idx.cpu(), model_indices_t[a2_idx.cpu()]]
    w_n = inclusion_probabilities[n_idx.cpu(), model_indices_t[a2_idx.cpu()]]
    # TODO: Checked only for TripletLoss, maybe 1+w should be placed somewhere else if it works only with this loss
    # TODO: Consider exposing following as option
    weights_matrix[a2_idx.cpu(), n_idx.cpu()] = 1 - w_n  # w_a2 * w_n + 1

    # for a_i, n_i in zip(a2_idx, n_idx):
    #     m_a = model_indices_t[a_i]
    #     w_a = inclusion_probabilities[a_i][m_a]
    #     # We check against the model of the anchor
    #     w_n = inclusion_probabilities[n_i][m_a]
    #     assert weights_matrix[a_i, n_i] == 1 + (w_a * w_n)

    return weights_matrix


def get_population_aware_pairs_indices(labels, ref_labels=None,
                                       rejection_strategy_object: RejectionStrategy = None):
    """
    Overrides the default get_all_pairs_indices, such as some indices are discarded according to the pa decision logic

    This is meant to be used with the help of partial function which has rejection_strategy_object bound to a specific
    object such that it can act as a drop-in replacement for get_all_pairs_indices.
    """

    # Initially, call the function which returns all the indices
    a1_idx, p_idx, a2_idx, n_idx = get_all_pairs_indices(labels, ref_labels)

    # If population aware approach is not preferred or has not yet initialized I simply return all the indices
    if rejection_strategy_object is None or not rejection_strategy_object.has_trained():
        return a1_idx, p_idx, a2_idx, n_idx

    if rejection_strategy_object.use_raw_probabilities:
        # Rejection in this case consists of deferring to the loss computation, and more specifically to distance
        # matrix weighting. The weights computed below will be used to create the NxN weight matrix.
        indices_tuple = (a1_idx, p_idx, a2_idx, n_idx)
        weights_matrix = sample_weights_impl(rejection_strategy_object=rejection_strategy_object,
                                             labels=labels, indices_tuple=indices_tuple)
        rejection_strategy_object.store_batch_weights(weights_matrix)
    else:
        # Overriden (reduced) indices - rejected as noise were removed
        a1_idx, p_idx, a2_idx, n_idx = \
            filter_indices_tuple(indices_tuple=(a1_idx, p_idx, a2_idx, n_idx),
                                 batch_noise_predictions=rejection_strategy_object.current_batch_noisy_predictions,
                                 noisy_positive_as_negative=rejection_strategy_object.noisy_positive_as_negative)

    return a1_idx, p_idx, a2_idx, n_idx


def create_pa_miner(base: Type[BaseMiner], **instance_arguments):
    """
    Defines a class, base of which is given by an argument. This is to create same extension functionality of multiple
    same-api and yet similar base miner classes.
    Otherwise, we had to extend each base class separately or do other tricks.

    @param base: The base of the class to de defined
    @param instance_arguments: Keyword arguments for instance initialization
    @return: an instance of PopulationAwareMiner, having as base class the given param
    """

    class PopulationAwareMiner(base):
        """
        PopulationAwareMiner is as two-stage miner which contains (a) mining logic itself and (b) an arbitrary external
        miner, the one defined as base class.
        (a) weights some pairs of indices, both a-p and a-n, such that they are considered as "bad" according to the
        population aware logic
        (b) second-phase miner operates directly on weighted distance matrix rather than the original one, therefore it
        does not consider pairs that are discarded from the first phase
        """

        SUPPORTED_MINERS = \
            [
                miners.BatchEasyHardMiner,
                miners.BatchHardMiner,
                miners.UniformHistogramMiner,
                # miners.TripletMarginMiner,
                PassThroughMiner,
                PRISM,
            ]

        def __init__(self, use_pretrained=False, population_warm_up_epochs=0,
                     strategy: Literal['populations', 'clusters', 'knn'] = 'populations', training_samples_fraction=1.0,
                     approach: Literal['OpenMax', 'EVM'] = 'OpenMax', tail_size=0.25, use_raw_probabilities=False,
                     decision_threshold=0.5,
                     distance_type: Literal['euclidean', 'cosine'] = 'euclidean',
                     noise_handling_kwargs=None,
                     with_inspector=False, keep_only_good_samples=False, distance_kwargs=None, **kwargs):
            super().__init__(**kwargs)

            # Check supported miners
            assert any(map(lambda x: isinstance(self, x), self.SUPPORTED_MINERS))

            # We keep the original distance (functor) object for computing the first phase distances
            self.orig_distance = copy.deepcopy(self.distance)
            del self.distance
            # We override the distance member with the weighted distance object.
            self.distance = create_distance_patcher(type(self.orig_distance), **distance_kwargs)

            self._keep_only_good_samples = keep_only_good_samples
            self._use_pretrained = use_pretrained
            self.population_warm_up_epochs = population_warm_up_epochs
            self.current_epoch = 0

            self._memory_bank = MemoryBank()

            self._inspector = Inspector() if with_inspector else None

            # TODO: Below I always set use raw probabilities because pair rejection expects show
            if strategy == 'populations':
                if noise_handling_kwargs is None:
                    noise_handling_options = NoiseHandlingOptions(nn_weights=None, density_mr=None)
                else:
                    nn_weights = NearestNeighborsWeightsOptions(**noise_handling_kwargs['nn_weights']) \
                        if 'nn_weights' in noise_handling_kwargs else None
                    density_mr = DensityBasedModelsReductionOptions(**noise_handling_kwargs['density_mr']) \
                        if 'density_mr' in noise_handling_kwargs else None

                    noise_handling_options = NoiseHandlingOptions(nn_weights=nn_weights, density_mr=density_mr)

                # PopulationAwarePairRejection object instantiation
                param_string = f'--distance_metric {distance_type} --tailsize {tail_size}'
                self.pair_rejection_strategy = \
                    PopulationAwarePairRejection(
                        approach=approach,
                        training_samples_fraction=training_samples_fraction,
                        rejection_criteria=HighScoreInPositiveClassCriterion(threshold=decision_threshold),
                        algorithm_parameters_string=param_string,
                        use_raw_probabilities=use_raw_probabilities,
                        noise_handling_options=noise_handling_options
                    )
            elif strategy == 'clusters':
                # ClusterAwarePairRejection object instantiation
                self.pair_rejection_strategy = \
                    ClusterAwarePairRejection(training_samples_fraction=training_samples_fraction)
            elif strategy == 'knn':
                self.pair_rejection_strategy = KNNPairRejection(**noise_handling_kwargs['nn_weights'],
                                                                training_samples_fraction=training_samples_fraction,
                                                                rejection_criteria=HighScoreInPositiveClassCriterion(
                                                                    threshold=decision_threshold),
                                                                use_raw_probabilities=use_raw_probabilities)
            else:
                raise RuntimeError(f'Unsupported strategy {strategy}')

            # In this case we override the indices function in order to intervene rejection
            sys.modules[
                'pytorch_metric_learning'].utils.loss_and_miner_utils.get_all_pairs_indices = \
                partial(get_population_aware_pairs_indices,
                        rejection_strategy_object=self.pair_rejection_strategy)

            # Initially set in mine() and afterwards used in distance(). Kind of spaghetti code to keep the same
            # class API and have minimal impact on
            # self.masker = None
            self.distance.weights = None

            # TODO: Consider addition of recordable attributes, that will be used for logging
            #  e.g. we can store the number of the remaining positives and negatives, in order to realise the progress of
            #  the miner across epochs

        def mine(self, embeddings, labels, ref_emb, ref_labels):
            """

            :param embeddings:
            :param labels:
            :param ref_emb:
            :param ref_labels:
            :param noisy_samples: Used only for debugging against ground truth
            :return:
            """
            # Call the actual distance function from the parent class
            mat = self.orig_distance(embeddings, ref_emb)

            dmax = mat.max()
            dmin = mat.min()

            if dmax - dmin < 1e-3:
                warn('Network has collapsed, both positive and negative samples are mapped (around) to single point')

            # We call the rejection step as soon as the trainer has been initialized
            if self.pair_rejection_strategy.has_trained():
                # OLD approach, using weighs on distance matrix
                # self.distance.weights = self.population_aware_rejection(embeddings, labels)

                self.pair_rejection_strategy.predict_noise(embeddings, labels)
                # NEW approach, pairs have been eliminated, therefore I mute weights with ones
                # TODO: I have kept old approach in comments in case it is useful later, need to distinguish
                #  between the two in a better way
                self.distance.weights = torch.ones_like(mat)
            else:
                # weights is full of ones therefore it has no effect when multiplied with distances matrix
                self.distance.weights = torch.ones_like(mat)

            # The mine function below internally calls the patched version of get_all_pairs_indices which excludes
            # pairs with noisy samples
            ret = super().mine(embeddings, labels, ref_emb, ref_labels)

            kept_embeddings = embeddings.clone().detach().to(dtype=torch.float16)
            kept_labels = labels.clone().detach()

            if self.pair_rejection_strategy.use_raw_probabilities and self._keep_only_good_samples:
                raise NotImplementedError
                # weights = self.pair_rejection_strategy.retrieve_batch_weights()
                # good_samples = weights > 0.5
                # kept_embeddings = kept_embeddings[good_samples]
                # kept_labels = kept_labels[good_samples]

            # Store current features
            self._memory_bank.add_to_memory(kept_embeddings, kept_labels)

            # Report if needed
            if self._inspector is not None:
                self._inspector.add_batch_info(batch_labels=labels,
                                               batch_predictions_scores=self._strategy.retrieve_batch_raw_scores(),
                                               batch_predictions_noisy=self._strategy.current_batch_noisy_predictions,
                                               batch_gt_noisy=self.current_batch_noisy_samples)

            return ret

        def output_assertion(self, output):
            """
            For now, I don't have any post mining assertion to put, therefore I override and leave the implementation empty

            TODO: Consider possible assertions, e.g. that there are enough triplets left for further loss computation, etc
            """
            super().output_assertion(output)

        def bootstrap_initial(self, embeddings, labels):
            """ Can be called before starting training to initialize with pretrained weights """
            print('Using vanilla features for training population aware rejection...')
            self._memory_bank.add_to_memory(embeddings, labels)
            self.pair_rejection_strategy.train(self._memory_bank)

        def bootstrap_epoch(self, epoch=-1):
            """
            This function should be called at the start of every epoch.

            @param epoch:
            @return:
            """
            # if self.loss_weight == 0:
            #    return

            if self.current_epoch >= self.population_warm_up_epochs:
                if self._inspector is not None:
                    self._inspector.report_and_reset(
                        self.pair_rejection_strategy.labels_to_indices)
                self.pair_rejection_strategy.train(self._memory_bank)
            else:
                self._memory_bank.reset_memory()

            if epoch == -1:
                self.current_epoch += 1
            else:
                self.current_epoch = epoch

        @property
        def inspector(self):
            return self._inspector

        @property
        def use_pretrained(self):
            return self._use_pretrained

        def forward(self, embeddings, labels, ref_emb=None, ref_labels=None, noisy_samples=None):
            self.pair_rejection_strategy.current_batch_noisy_samples = noisy_samples
            return super().forward(embeddings, labels, ref_emb, ref_labels)

    return PopulationAwareMiner(**instance_arguments)
