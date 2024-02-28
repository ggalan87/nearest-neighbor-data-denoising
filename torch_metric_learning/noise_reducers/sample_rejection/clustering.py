from typing import Dict

import numpy as np
import torch
from PEACH import PEACH as PEACH_Algo
from pytorch_metric_learning.distances import LpDistance
from torch.nn import functional as F

from lightning.losses_playground import ExclusionInspector
from torch_metric_learning.noise_reducers.memory_bank import MemoryBank
from torch_metric_learning.noise_reducers.sample_rejection.rejection_base import RejectionStrategy


class ClusterAwarePairRejection(RejectionStrategy):
    def __init__(self, training_samples_fraction=1.0, noisy_positive_as_negative=False):

        # With Cluster Aware Pair Rejection we do not have probabilities but instead only binary inclusions
        # TODO: Find a way to express inclusion with continuous value e.g. similarity
        super().__init__(training_samples_fraction, noisy_positive_as_negative, use_raw_probabilities=False)

        self._predicted_clusters = None
        self._cluster_centroids = None
        self._labels_to_clusters = None
        self._labels_to_indices = None

        self._distance = LpDistance(normalize_embeddings=True)

        self.ei = ExclusionInspector()

    def train(self, memory_bank: MemoryBank):
        all_features, all_class_labels, _, _ = memory_bank.get_data(self._training_samples_fraction)

        print('PEACH clustering...')

        # Internally PEACH uses a function which does not work with half precision resulting to error
        # RuntimeError: "median_cpu" not implemented for 'Half'
        all_features = all_features.to(dtype=torch.float32).cpu()
        with_evt = False
        no_singleton = True
        result = PEACH_Algo(all_features, 0, no_singleton=no_singleton, metric='cosine', batch_size=4096, evt=with_evt)
        clusters = torch.tensor(np.array(result))

        # Count clusters except invalids (cid = -1)
        # TODO: Find why not all samples are assigned to a label
        unique_clusters = torch.unique(clusters[clusters != -1])
        cluster_centroids = torch.zeros((len(unique_clusters), all_features.shape[1]))

        labels_to_clusters = {}
        labels_to_indices = {}

        assert clusters.max() == len(unique_clusters) - 1

        # Compute centroids
        for cluster_id in unique_clusters:
            c_indices = torch.where(clusters == cluster_id)[0]
            cluster_centroids[cluster_id] = torch.mean(all_features[c_indices], dim=0)

        # Find labels to clusters correspondences
        for i, l in enumerate(torch.unique(all_class_labels)):
            l_indices = torch.where(all_class_labels == l)[0]
            pred_for_l = clusters[l_indices]

            # print(np.count_nonzero(pred_for_l == -1) / len(pred_for_l))
            pred_for_l = pred_for_l[pred_for_l > -1]

            # pred_for_l += 1
            counts = torch.bincount(pred_for_l)
            # print(f'label {l} was spread into clusters {torch.unique(pred_for_l)},'
            #       f' but most of them ({torch.max(counts)}) were in {torch.argmax(counts)}')
            label_cluster = torch.argmax(counts)
            labels_to_clusters[int(l)] = label_cluster
            labels_to_indices[int(l)] = i

        # Assign to instance variables
        self._cluster_centroids = cluster_centroids.cuda()
        self._labels_to_clusters = labels_to_clusters
        self._labels_to_indices = labels_to_indices

    def has_trained(self):
        return self._cluster_centroids is not None

    def _store_batch_raw_scores(self, embeddings, labels, normalize=True):
        self._batch_predictions = self._strategy_impl(embeddings, labels)

    @property
    def labels_to_indices(self) -> Dict[int, int]:
        return self._labels_to_indices

    def _strategy_impl(self, embeddings, labels) -> torch.Tensor:
        inclusion_labels = torch.zeros((len(embeddings), len(self.labels_to_indices)))
        normalized_embeddings = F.normalize(embeddings)

        distances = self._distance(normalized_embeddings, self._cluster_centroids)

        predicted_clusters = torch.argmax(distances, dim=1)

        for i, l in enumerate(labels):
            if predicted_clusters[i] == self._labels_to_clusters[int(l)]:
                inclusion_labels[i] = 1.0

        return inclusion_labels
