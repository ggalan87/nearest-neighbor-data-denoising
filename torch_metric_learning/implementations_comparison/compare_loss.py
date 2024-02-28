from pathlib import Path
import torch
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f

# Common things
margin = 0.2
distance_func = distances.LpDistance()  # Euclidean distance


def pytorch_metric_learning_triplet_margin_loss(embeddings, labels, indices_tuple):
    c_f.labels_or_indices_tuple_required(labels, indices_tuple)
    indices_tuple = lmu.convert_to_triplets(
        indices_tuple, labels, labels, t_per_anchor='all'
    )
    anchor_idx, positive_idx, negative_idx = indices_tuple
    if len(anchor_idx) == 0:
        return None

    mat = distance_func(embeddings, embeddings)
    ap_dists = mat[anchor_idx, positive_idx]
    an_dists = mat[anchor_idx, negative_idx]

    current_margins = distance_func.margin(ap_dists, an_dists)
    violation = current_margins + margin

    loss = torch.nn.functional.relu(violation)

    # reduction
    loss = torch.mean(loss[loss > 0])
    return loss


def torchreid_triplet_margin_loss(embeddings, labels, indices_tuple):
    mat = distance_func(embeddings, embeddings)

    anchor_idx, positive_idx, negative_idx = indices_tuple

    ap_dists = mat[anchor_idx, positive_idx]
    an_dists = mat[anchor_idx, negative_idx]

    y = torch.ones_like(an_dists)

    ranking_loss = torch.nn.MarginRankingLoss(margin=margin)
    loss = ranking_loss(an_dists, ap_dists, y)
    return loss


if __name__ == '__main_':
    anchors = torch.load('anchors.pt')
    positives = torch.load('positives.pt')
    negatives = torch.load('negatives.pt')
    embeddings = torch.load('embeddings.pt')
    labels = torch.load('labels.pt')

    loss = pytorch_metric_learning_triplet_margin_loss(embeddings, labels, (anchors, positives, negatives))
    loss2 = torchreid_triplet_margin_loss(embeddings, labels, (anchors, positives, negatives))
    print(loss, loss2)
