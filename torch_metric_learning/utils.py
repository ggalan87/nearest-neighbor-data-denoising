import copy
from typing import Type, Optional, Dict
from warnings import warn

import torch
from pytorch_metric_learning import distances
from pytorch_metric_learning.utils.inference import CustomKNN


def create_distance_patcher(base: Type[distances.BaseDistance], **instance_arguments):
    """
    Defines a class, base of which is given by an argument. This is to create same extension functionality of multiple
    same-api and yet similar base classes whose role is to compute distances.
    Otherwise, we had to extend each base class separately or do other tricks.

    @param base: The base of the class to de defined
    @param instance_arguments: Keyword arguments for instance initialization
    @return: an instance of DistancePatch, having as base class the given param
    """

    class DistancePatch(base):
        """
        DistancePatch is an extension to arbitrary base classes from pytorch_metric_learning distance module. Here we
        simply override th compute_mat() method which is the method which actually computes the distance matrix, such
        that it computes a weighted distance matrix. This is in line with Population Aware algorithm which aims to
        discard pairs of embeddings. Weighting is computed such that a miner which is called afterwards discards the
        under-weighted pairs.

        This is achieved by using a masker instance. The masker is same size as the distance
        matrix that is computed, having all indices equal to 1, except the indices that need to be discarded. These
        indices are set to minimum or maximum values that are contrary to the mining technique which will be followed
        afterwards. For example in the case of hard positive and hard negative mining, we do the following.
        Let say, a,p is a pair that we want to discard. The element of the masker matrix which corresponds to a-p
        distance is set to a minimum value e.g. 0 such that if multiplied with the real distance, this will be
        "discarded". Let say a,n is another pair we want to discard. We correspondingly set the masker element to a
        large number, just to ensure that the element will never be selected as hard negative.

        We follow such approach and don't operate directly on the distance matrix in order to decouple the computations.
        PS: In an earlier implementation this was also mandatory for being able to intervene into the computational
        graph, however, the current implementation uses detached variables, and it is not mandatory.
        This is because assignment is not differentiable, however multiplication is. Both have the same effect in mining.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Initialize with one such that multiplication is identity
            # TODO: Get device from object
            self._weights = torch.tensor([1]).to('cuda:0')

        def compute_mat(self, query_emb, ref_emb):
            # Check for inverted distance (similarity) in order to invert the weighting

            # TODO: Fix the following
            warn('Distance is inverted check is muted (hardcoded)')
            # if self.is_inverted:
            #     raise NotImplementedError
            #
            #     # w = 1 - self.weights if self.is_inverted else self.weights

            w = self.weights

            d = super().compute_mat(query_emb, ref_emb)
            n, m = d.shape

            if n != m:
                raise NotImplementedError

            # check if w is "vector" of size (n,) or (n,1)
            # if w.shape[0] == n and w.numel() == n:
            #     return w.repeat_interleave(m).reshape(n, m).to('cuda:0') * d

            return w.to('cuda:0') * d

        @property
        def weights(self):
            return self._weights

        @weights.setter
        def weights(self, val):
            self._weights = val

    return DistancePatch(**instance_arguments)


def patch_object_with_distance(object_with_distance, distance_kwargs: Optional[Dict] = None):
    orig_distance = copy.deepcopy(object_with_distance.distance)
    del object_with_distance.distance
    # We override the distance member with the weighted distance object.

    if distance_kwargs is None:
        distance_kwargs = {}
    object_with_distance.distance = create_distance_patcher(type(orig_distance), **distance_kwargs)


class BatchedKNN(CustomKNN):
    def __init__(self, distance, batch_size=None):
        super().__init__(distance, batch_size)
        self._embeddings = None

    def train(self, embeddings):
        self._embeddings = embeddings

    def __call__(self, query, k, reference=None, ref_includes_query=False):
        return super(BatchedKNN, self).__call__(query, k, self._embeddings)
