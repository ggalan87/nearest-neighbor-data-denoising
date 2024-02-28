from typing import Type
import torch
from pytorch_metric_learning.reducers import BaseReducer, AvgNonZeroReducer
import pytorch_metric_learning.utils.loss_and_miner_utils as lmu


def create_noisy_weighted_reducer(base: Type[BaseReducer], **instance_arguments):
    class NoisyWeightedReducer(base):

        SUPPORTED_REDUCERS = \
            [
                AvgNonZeroReducer
            ]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Check supported reducers
            assert any(map(lambda x: isinstance(self, x), self.SUPPORTED_REDUCERS))

            # TODO: Get device from object
            self._weights = torch.tensor([1]).to('cuda:0')
            self._mined_indices = None

        def element_reduction(self, losses, loss_indices, embeddings, labels):
            raise NotImplementedError
            # return super().element_reduction(losses, loss_indices, embeddings, labels)

        def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
            raise NotImplementedError
            # return super().pos_pair_reduction(losses, loss_indices, embeddings, labels)

        def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
            raise NotImplementedError
            # return super().neg_pair_reduction(losses, loss_indices, embeddings, labels)

        def triplet_reduction(self, losses, loss_indices, embeddings, labels):
            if len(self._weights) != 1:
                anchor_idx, positive_idx, negative_idx = lmu.convert_to_triplets(
                    self.mined_indices, labels, ref_labels=None, t_per_anchor='all'
                )

                ap_weights = self._weights[anchor_idx, positive_idx]
                n_weights = self._weights[anchor_idx, negative_idx]
                losses = torch.sigmoid(ap_weights * n_weights) * losses

            return super().triplet_reduction(losses, loss_indices, embeddings, labels)

        @property
        def weights(self):
            return self._weights

        @weights.setter
        def weights(self, val):
            self._weights = val

        @property
        def mined_indices(self):
            return self._mined_indices

        @mined_indices.setter
        def mined_indices(self, val):
            self._mined_indices = val

    return NoisyWeightedReducer(**instance_arguments)
