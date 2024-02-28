import torch
from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.losses import CrossBatchMemory
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class PRISM(BaseMiner):
    def __init__(self, num_classes: int, cross_batch_memory_object: CrossBatchMemory, noise_rate=0.5, window_size=30,
                 **kwargs):
        super(PRISM, self).__init__(**kwargs)
        self._xbm_object = cross_batch_memory_object
        # The centroids
        self._centroids = torch.zeros(size=[num_classes, self._xbm_object.embedding_size]).cuda()
        # Set of class ids
        self._filled_center = set()

        self._last_clean_labels = None

        self._noise_rate = noise_rate
        self._window_size = window_size
        self._margin_window = []

        self._clean_in_batch = None

    def _get_mem(self):
        if not self._xbm_object.has_been_filled:
            return self._xbm_object.embedding_memory[: self._xbm_object.queue_idx], \
                self._xbm_object.label_memory[: self._xbm_object.queue_idx]

        return self._xbm_object.embedding_memory, self._xbm_object.label_memory

    def _update_centroids(self):
        if self._last_clean_labels is None:
            return

        features, targets = self._get_mem()

        for i in self._last_clean_labels:
            i = i.item()
            row_mask = (targets == i)
            self._centroids[i] = torch.mean(features[row_mask], dim=0)
            self._filled_center.add(i)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        # Ensure normalized, because similarity below assumes so
        embeddings = self.distance.maybe_normalize(embeddings)

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
        idx_sorted = torch.argsort(C)
        to_remove = idx_sorted[:int(self._noise_rate * len(C))]

        # update window
        if not torch.isnan(C[to_remove[-1]]) and C[to_remove[-1]] != 1.0:
            self._margin_window.append(C[to_remove[-1]].item())
            self._margin_window = self._margin_window[-self._window_size:]

        if len(self._margin_window) > 0:
            keep_bool = (C > sum(self._margin_window) / len(self._margin_window))
            if torch.any(keep_bool) == False:
                keep_bool = torch.ones_like(labels, dtype=bool)
                keep_bool[to_remove] = False
                self._margin_window = self._margin_window[-1:]
            else:
                pass
        else:
            keep_bool = torch.ones_like(labels, dtype=bool)
            keep_bool[to_remove] = False

        clean_labels = labels[keep_bool]
        self._last_clean_labels = torch.unique(clean_labels)

        self._clean_in_batch = keep_bool

        # Having computed the mask, we now use it to create required tuple fore return using only the reduced labels
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(clean_labels)
        return a1_idx, p_idx, a2_idx, n_idx
