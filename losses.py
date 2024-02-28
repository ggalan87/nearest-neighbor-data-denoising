"""
File from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/losses/hard_mine_triplet_loss.py
"""

from __future__ import division, absolute_import
import torch
import torch.nn as nn
import torch.functional as F

from mmcls.models.builder import LOSSES

# from mmdet.models.losses.utils import weighted_loss
#
# @weighted_loss
# def my_loss(pred, target):
#     assert pred.size() == target.size() and target.numel() > 0
#     loss = torch.abs(pred - target)
#     return loss

def jaccard_cont_torch(x1, x2):
    x1_sig = torch.sigmoid(x1)
    x2_sig = torch.sigmoid(x2)

    sim = torch.divide(torch.sum(torch.minimum(x1_sig, x2_sig)),
                        torch.sum(torch.maximum(x1_sig, x2_sig)))
    return 100 * torch.subtract(torch.ones_like(sim), sim)


def pairwise_euclidean(inputs):
    n = inputs.size(0)

    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    return dist


def pairwise_jaccard(inputs):
    n = inputs.size(0)
    dist = torch.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist[i, j] = dist[j, i] = jaccard_cont_torch(inputs[i], inputs[j])

    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    return dist

@LOSSES.register_module()
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, loss_weight=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.loss_weight = loss_weight

    def forward(self, inputs, targets, avg_factor=None):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        dist = pairwise_euclidean(inputs)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return self.loss_weight * loss

    # def backward(self):
    #     pass


@LOSSES.register_module()
class QuadrupletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, loss_weight=1.0, super_classes=[[]]):
        super(QuadrupletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_class = nn.MarginRankingLoss(margin=margin)
        self.loss_weight = loss_weight

        self.super_classes = super_classes

        self.id2c = [-1] * sum([len(sc) for sc in self.super_classes])
        for i, c in enumerate(self.super_classes):
            for id in c:
                self.id2c[id - 1] = i

        self.id2c = torch.Tensor(self.id2c).cuda()

    def forward(self, inputs, targets, avg_factor=None):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        batch_classes = torch.zeros_like(targets)
        for i, t in enumerate(targets):
            batch_classes[i] = self.id2c[t - 1]  # ids start from 1

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)

        # dist = torch.pdist(inputs)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        #print(mask)

        mask_ids = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_classes = batch_classes.expand(n, n).eq(batch_classes.expand(n, n).t())

        mask_neg_ids = torch.logical_xor(mask_ids, mask_classes)
        mask_neg_classes = torch.logical_not(mask_classes)

        dist_ap, dist_ani, dist_anc = [], [], []
        for i in range(n):
            dist_ap.append(dist[i][mask_ids[i]].max().unsqueeze(0))
            dist_ani.append(dist[i][mask_neg_ids[i]].min().unsqueeze(0))
            dist_anc.append(dist[i][mask_neg_classes[i]].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_ani = torch.cat(dist_ani)
        dist_anc = torch.cat(dist_anc)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_ani)
        loss = self.ranking_loss(dist_ani, dist_ap, y) + self.ranking_loss_class(dist_anc, dist_ap, y)
        return self.loss_weight * loss

    # def backward(self):
    #     pass


@LOSSES.register_module()
class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Adapted from: https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch

        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

