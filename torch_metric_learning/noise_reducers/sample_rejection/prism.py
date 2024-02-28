from warnings import warn
from typing import Dict
import torch
import torch.nn.functional as F
from torch_metric_learning.noise_reducers.memory_bank import MemoryBank
from torch_metric_learning.noise_reducers.sample_rejection import RejectionStrategy
from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import CombinedCriteria


class PRISM(RejectionStrategy):
    def __init__(self, num_classes: int, noise_rate=0.5, window_size=30):
        super().__init__(training_samples_fraction=1.0,
                         # dummy criterion -> combined object with empty list / never called
                         rejection_criteria=CombinedCriteria([]),
                         noisy_positive_as_negative=False,
                         use_raw_probabilities=False)

        self._memory_bank = None
        self._labels_to_indices = None
        self._num_classes = num_classes

        # The centroids
        self._centroids = None
        # Set of class ids
        self._filled_center = set()

        self._last_clean_labels = None

        self._noise_rate = noise_rate
        self._window_size = window_size
        self._margin_window = []

        self._clean_in_batch = None

    def train(self, memory_bank: MemoryBank):
        # Training consists of only setting the memory bank, scores are computed then directly using this bank
        # The memory is set once, because the instance does not change
        if memory_bank.with_dynamic_memory:
            raise AssertionError('PRISM should be used with preallocated memory')
        elif self._memory_bank is not None:
            # Memory bank object is the same, also centroids need to be initialized once and afterwards only update
            return

        self._memory_bank = memory_bank

        xbm_object = self._memory_bank.mem_object.memory
        self._centroids = torch.zeros(size=[self._num_classes, xbm_object.embedding_size]).cuda()

    def has_trained(self):
        return self._memory_bank is not None

    def _store_batch_raw_scores(self, embeddings, labels, normalize=True):
        # Ensure normalized, because similarity below assumes so
        embeddings = F.normalize(embeddings)

        self._update_centroids()

        C = []
        for i, target in enumerate(labels):
            selected_cls = target.item()
            if selected_cls not in self._filled_center:
                C.append(1)
            else:
                all_sim_exp = torch.exp(torch.mm(self._centroids, embeddings[i].view(-1, 1))).view(-1)
                softmax_loss = all_sim_exp[selected_cls] / torch.sum(all_sim_exp)
                C.append(softmax_loss)

        C = torch.Tensor(C)

        # TODO: move rest logic into _compute_noise_predictions as this is the case for the function
        #  on the other hand cannot set here the raw scores because for the other strategies it is a
        #  NxC matrix, here is Nx1
        idx_sorted = torch.argsort(C)
        to_remove = idx_sorted[:int(self._noise_rate * len(C))]

        # update window
        if not torch.isnan(C[to_remove[-1]]) and C[to_remove[-1]] != 1.0:
            self._margin_window.append(C[to_remove[-1]].item())
            self._margin_window = self._margin_window[-self._window_size:]

        if len(self._margin_window) > 0:
            keep_bool = (C > sum(self._margin_window) / len(self._margin_window))
            if torch.any(keep_bool) == False:
                keep_bool = torch.ones_like(labels, dtype=torch.bool)
                keep_bool[to_remove] = False
                self._margin_window = self._margin_window[-1:]
            else:
                pass
        else:
            keep_bool = torch.ones_like(labels, dtype=torch.bool)
            keep_bool[to_remove] = False

        clean_labels = labels[keep_bool]
        self._last_clean_labels = torch.unique(clean_labels)

        self._clean_in_batch = keep_bool

    def _compute_noise_predictions(self, labels):
        """ I override because PRISM does not work with the rejection strategies """
        batch_size = len(labels)

        # Initialize with all False (clean), and then pass the samples to see which are noisy
        self._batch_noise_predictions = torch.zeros((batch_size,), dtype=torch.bool)

        # I keep it verbose
        for i in range(batch_size):
            self._batch_noise_predictions[i] = not self._clean_in_batch[i]

    @property
    def labels_to_indices(self) -> Dict[int, int]:
        warn('Labels to indices cannot be properly computed and be 100% sure, because even when memory is filled'
             'some labels may have been entirely omitted as noisy')
        return {}

    def _update_centroids(self):
        if self._last_clean_labels is None:
            return

        features, targets, _, _ = self._memory_bank.get_data(samples_fraction=1.0, do_reset_memory=False)

        for i in self._last_clean_labels:
            i = i.item()
            row_mask = (targets == i)
            self._centroids[i] = torch.mean(features[row_mask], dim=0)
            self._filled_center.add(i)
