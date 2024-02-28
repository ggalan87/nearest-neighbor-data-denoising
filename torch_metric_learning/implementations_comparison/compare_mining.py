from pathlib import Path
from horology import Timing
import torch
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f

# Common things
margin = 0.2

# This includes L2 normalization by default
distance_func = distances.LpDistance()  # Euclidean distance



def pytorch_metric_learning_triplet_margin_miner(embeddings, labels):
    anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(
        labels, labels
    )

    mat = distance_func(embeddings, embeddings)
    ap_dist = mat[anchor_idx, positive_idx]
    an_dist = mat[anchor_idx, negative_idx]

    triplet_margin = (
        ap_dist - an_dist if distance_func.is_inverted else an_dist - ap_dist
    )

    # hard mining
    threshold_condition = triplet_margin <= margin
    threshold_condition &= triplet_margin <= 0

    a_idx = anchor_idx[threshold_condition]
    p_idx = positive_idx[threshold_condition]
    n_idx = negative_idx[threshold_condition]
    return a_idx, p_idx, n_idx


def get_x_per_row(
        xtype,
        mat,
        anchor_idx,
        other_idx,
        val_range=None,
        semihard_thresholds=None,
):
    assert xtype in ["min", "max"]
    inf = c_f.pos_inf(mat.dtype) if xtype == "min" else c_f.neg_inf(mat.dtype)
    mask = torch.ones_like(mat) * inf
    mask[anchor_idx, other_idx] = 1
    # if semihard_thresholds is not None:
    #     if xtype == "min":
    #         condition = mat <= semihard_thresholds.unsqueeze(1)
    #     else:
    #         condition = mat >= semihard_thresholds.unsqueeze(1)
    #     mask[condition] = inf
    # if val_range is not None:
    #     mask[(mat > val_range[1]) | (mat < val_range[0])] = inf

    non_inf_rows = torch.any(mask != inf, dim=1)
    mat = mat.clone()
    mat[mask == inf] = inf
    dist_fn = torch.min if xtype == "min" else torch.max
    return dist_fn(mat, dim=1), non_inf_rows


def pytorch_metric_learning_batch_hard_miner(embeddings, labels):
    def get_positives(mat, a1_idx, p_idx, negative_dists=None):
        return get_x_per_row("max", mat, a1_idx, p_idx, None, negative_dists)

    def get_negatives(mat, a2_idx, n_idx, positive_dists=None):
        return get_x_per_row("min", mat, a2_idx, n_idx, None, positive_dists)

    mat = distance_func(embeddings, embeddings)
    a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels, labels)
    a = torch.arange(mat.size(0), device=mat.device)

    (positive_dists, positive_indices), a1p_keep = get_positives(mat, a1_idx, p_idx)
    (negative_dists, negative_indices), a2n_keep = get_negatives(mat, a2_idx, n_idx)

    a_keep_idx = torch.where(a1p_keep & a2n_keep)
    a = a[a_keep_idx]
    p = positive_indices[a_keep_idx]
    n = negative_indices[a_keep_idx]
    return a, p, n


def torchreid_batch_hard_miner(embeddings, labels):
    n = embeddings.size(0)
    with Timing(name='Distance: '):
        dist = distance_func(embeddings, embeddings)

    # For each anchor, find the hardest positive and negative
    mask = labels.expand(n, n).eq(labels.expand(n, n).t())
    ap, an = [], []

    with Timing(name='Important calculations: '):
        for i in range(n):
            dist_i = torch.clone(dist[i])
            dist_i[mask[i] == 0] = float('-inf')
            ap.append(dist_i.argmax().unsqueeze(0))

            dist_i = torch.clone(dist[i])
            dist_i[mask[i]] = float('inf')
            an.append(dist_i.argmin().unsqueeze(0))

    a = torch.arange(embeddings.size(0), device=embeddings.device)
    ap = torch.cat(ap)
    an = torch.cat(an)

    return a, ap, an

embeddings = torch.load('embeddings.pt')
labels = torch.load('labels.pt')
a1, ap1, an1 = pytorch_metric_learning_batch_hard_miner(embeddings, labels)
a2, ap2, an2 = torchreid_batch_hard_miner(embeddings, labels)
print('test')

# if __name__ == '__main_':
#     embeddings = torch.load('embeddings.pt')
#     labels = torch.load('labels.pt')
#     a1, ap1, an1 = pytorch_metric_learning_batch_hard_miner(embeddings, labels)
#     a2, ap2, an2 = torchreid_batch_hard_miner(embeddings, labels)
#     print('test')
#     # torch.save(a1, 'anchors.pt')
#     # torch.save(ap1, 'positives.pt')
#     # torch.save(an1, 'negatives.pt')
#
#     pass