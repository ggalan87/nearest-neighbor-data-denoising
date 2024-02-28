import torch
from torch import nn

""" Modified from PRISM paper, excluding configuration """

log_info = dict()


class XBM:
    """
    This class is from the original XBM paper "Cross-Batch Memory for Embedding Learning (XBM)"
    """
    def __init__(self, memory_size, features_dim):
        self.K = memory_size
        self.feats = torch.zeros(self.K, features_dim).cuda()
        self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.targets[:] = -1
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size


class PrismXBM(XBM):
    """
    This is an enhances XBM with functionality required by PRISM, moved from the loss module
    """
    def __init__(self, memory_size, features_dim, num_classes):
        super().__init__(memory_size=memory_size, features_dim=features_dim)
        # The centroids
        self._center = torch.zeros(size=[num_classes, features_dim]).cuda()
        # Set of class ids
        self._filled_center = set()

    def _update_center(self, targets):
        """

        :param new_targets: Unique targets that were added recently
        :return:
        """

        # Get only the unique
        new_targets = torch.unique(targets)

        features, targets = self.get()

        for i in new_targets:
            i = i.item()
            row_mask = (targets == i)
            self._center[i] = torch.mean(features[row_mask], dim=0)
            self._filled_center.add(i)

    def enqueue_dequeue(self, feats, targets):
        """
        Extends the parent impl in order to also update centers after insertion of new features and corresponding
        targets.

        :param feats:
        :param targets:
        :return:
        """
        super().enqueue_dequeue(feats, targets)
        self._update_center(targets)

    def target_is_in_memory(self, target):
        return target in self._filled_center

    @property
    def center(self):
        return self._center


class MemoryContrastiveLossPRISM(nn.Module):
    def __init__(self, num_classes, features_dim, memory_size, noise_rate=0.5, noise_window_size=30, noise_warm_up=0):
        super(MemoryContrastiveLossPRISM, self).__init__()
        self.margin = 0.5
        self.noise_rate = noise_rate
        self.last_target_col = None
        self.margin_window = []
        self.window_size = int(noise_window_size)
        self.iteration = 0
        self.start_check_noise_iteration = noise_warm_up

        self._xbm = PrismXBM(memory_size=memory_size, features_dim=features_dim, num_classes=num_classes)

    def _prism_operation(self, features, targets):
        with torch.no_grad():
            C = []
            for i, target in enumerate(targets):
                selected_cls = target.item()
                if not self._xbm.target_is_in_memory(selected_cls):
                    C.append(1)
                else:
                    all_sim_exp = torch.exp(torch.mm(self._xbm.center, features[i].view(-1, 1))).view(-1)
                    softmax_loss = all_sim_exp[selected_cls] / torch.sum(all_sim_exp)
                    C.append(softmax_loss)

            C = torch.Tensor(C)
            idx_sorted = torch.argsort(C)
            to_remove = idx_sorted[:int(self.noise_rate * len(C))]

            # update window
            if not torch.isnan(C[to_remove[-1]]) and C[to_remove[-1]] != 1.0:
                self.margin_window.append(C[to_remove[-1]].item())
                self.margin_window = self.margin_window[-self.window_size:]

            if len(self.margin_window) > 0:
                keep_bool = (C > sum(self.margin_window) / len(self.margin_window))
                if torch.any(keep_bool) == False:
                    keep_bool = torch.ones_like(targets, dtype=bool)
                    keep_bool[to_remove] = False
                    self.margin_window = self.margin_window[-1:]
                    # log_info[f"PRISM_threshold"] = C[to_remove[-1]]
                else:
                    # log_info[f"PRISM_threshold"] = sum(self.margin_window) / len(self.margin_window)
                    pass
            else:
                keep_bool = torch.ones_like(targets, dtype=bool)
                keep_bool[to_remove] = False
                # log_info[f"PRISM_threshold"] = C[to_remove[-1]]

        return keep_bool

    def forward(self, inputs_col, targets_col, inputs_row, target_row, is_noise=None):
        n = inputs_col.size(0)
        if inputs_row.shape[0] == 0:
            return 0, torch.ones_like(targets_col, dtype=bool).cuda()
        is_batch = (targets_col is target_row)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        epsilon = 1e-5
        loss = list()

        neg_count = list()

        pos_mask = targets_col.expand(target_row.shape[0], n).t() == target_row.expand(n, target_row.shape[0])
        neg_mask = (~pos_mask) & (sim_mat > self.margin)
        pos_mask = pos_mask & (sim_mat < (1 - epsilon))

        pos_pair = sim_mat[pos_mask]
        neg_pair = sim_mat[neg_mask]

        if not is_batch:
            if self.iteration < self.start_check_noise_iteration:
                keep_bool = torch.ones_like(targets_col, dtype=bool)
            else:
                keep_bool = self._prism_operation(features=inputs_col, targets=targets_col)

                for i in range(len(keep_bool)):
                    assert len(keep_bool) == len(pos_mask)
                    if keep_bool[i] == False:
                        pos_mask[i, :] = False
                        neg_mask[i, :] = False

                pos_pair = sim_mat[pos_mask]
                neg_pair = sim_mat[neg_mask]
        else:
            self.iteration += 1

        # if is_noise is not None:
        #     is_noise = is_noise[keep_bool]
        #     log_info[f"PRISM_pure_rate"] = 1 - torch.mean(is_noise.float()).item()

        pos_loss = torch.sum(-pos_pair + 1)
        if len(neg_pair) > 0:
            neg_loss = torch.sum(neg_pair)
        else:
            neg_loss = 0

        if is_batch:
            prefix = "batch_"
        else:
            prefix = "memory_"

        # log_info[f"{prefix}non_zero"] = len(neg_count)

        loss = (pos_loss + neg_loss) / n  # / all_targets.shape[1]

        if not is_batch:
            # log_info[f"xbm_loss"] = loss.item()
            # log_info[f"PRISM_remove_ratio"] = torch.mean(keep_bool.float()).item()
            return loss, keep_bool
        else:
            return loss

    @property
    def xbm(self):
        return self._xbm
