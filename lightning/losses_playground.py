from warnings import warn
from collections import namedtuple, defaultdict
import torch
import torch.nn as nn
import torch.functional as F

from evt.vast_openset import *
from evt.vast_ext import filter_labels
from features_storage import FeaturesStorage

SampleInfo = namedtuple('SampleInfo', ['klass', 'index'])


def pairwise_euclidean(inputs):
    n = inputs.size(0)

    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    return dist


class ExclusionInspector:
    def __init__(self):
        self.hard_positive_samples = defaultdict(list)
        self.hard_negative_samples = defaultdict(list)
        self.positives_excluded_per_iteration = list()
        self.iteration = 0

    def reset_store(self):
        self.hard_positive_samples = defaultdict(list)
        self.hard_negative_samples = defaultdict(list)
        self.positives_excluded_per_iteration = list()
        self.iteration = 0

    def add_hard_positive(self, sample_class, sample_index):
        self.hard_positive_samples[self.iteration].append(SampleInfo(klass=sample_class, index=sample_index))

    def add_hard_negative(self, sample_class, sample_index):
        self.hard_negative_samples[self.iteration].append(SampleInfo(klass=sample_class, index=sample_index))

    def add_positives_excluded_info(self, n_included, n_excluded):
        # TODO: Buggy
        # print(n_included, n_excluded)
        self.positives_excluded_per_iteration.append((int(n_included), int(n_excluded)))

    def report(self):
        """
        The motivation for the report is that as the learning progresses, the number of hard positive and hard negative
        will reduce because the model should have learned to map input into more distinct areas in the embedding space

        @return:
        """
        print('\n*********** Exclusion Inspector Report ***********')

        # n_iterations = len(self.hard_positive_samples)
        # n_hard_positive = 0
        # n_hard_negative = 0
        #
        # for it, samples in self.hard_positive_samples.items():
        #     # print(it, len(samples))
        #     n_hard_positive += len(samples)
        # for it, samples in self.hard_negative_samples.items():
        #     # print(it, len(samples))
        #     n_hard_negative += len(samples)
        #
        # if n_iterations != 0:
        #     print(f'Average hard positives per iteration: {n_hard_positive / n_iterations}')
        #     print(f'Average hard negatives per iteration: {n_hard_negative / n_iterations}')
        # else:
        #     print(f'Zero iterations')
        n_iterations = len(self.positives_excluded_per_iteration)

        if n_iterations == 0:
            print(f'Zero iterations')
            return

        ratios = []
        for (incl, excl) in self.positives_excluded_per_iteration:
            ratios.append(incl / (incl + excl))

        print(f'Avg good samples ratio {np.mean(np.array(ratios))}')

    def advance_iteration(self):
        self.iteration += 1


class PopulationAwareTripletLoss(nn.Module):
    """
    Population-aware batch hard sampling for triplet loss is an extension to triplet loss with batch hard sampling
    by additionally introducing global information about the population of the class

    https://stats.stackexchange.com/questions/475655/in-training-a-triplet-network-i-first-have-a-solid-drop-in-loss-but-eventually
    """

    def __init__(self, margin=0.3, loss_weight=1.0, loss_warm_up_epochs=0, semi_hard_warm_up_epochs=0, population_warm_up_epochs=0):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.loss_weight = loss_weight

        # Specify algorithm parameters for openset
        approach = 'EVM'
        algorithm_parameters = "--distance_metric euclidean --tailsize 1.0"

        if approach == 'EVM':
            algorithm_parameters += " --distance_multiplier 0.7"

        saver_parameters = f"--OOD_Algo {approach}"
        self.openset_model_params = OpensetModelParameters(approach, algorithm_parameters, saver_parameters)

        self.trainer = None
        self.current_epoch = 0
        self.loss_warm_up_epochs = loss_warm_up_epochs
        self.semi_hard_warm_up_epochs = semi_hard_warm_up_epochs
        self.population_warm_up_epochs = population_warm_up_epochs

        self.feats_list = []
        self.labels_list = []

        self.ei = ExclusionInspector()

    def bootstrap_epoch(self, epoch=-1):
        """
        This function should be called at the start of every epoch.

        @param epoch:
        @return:
        """
        if self.loss_weight == 0:
            return

        if self.current_epoch >= self.population_warm_up_epochs:
            all_features = torch.vstack(self.feats_list)
            all_class_labels = torch.hstack(self.labels_list)

            self.feats_list = []
            self.labels_list = []

            training_feats = OpensetData(features=all_features.to(torch.float32), class_labels=all_class_labels)
            self.trainer = OpensetTrainer(training_feats, self.openset_model_params)
            print('Openset training')
            self.trainer.train()
        else:
            self.feats_list = []
            self.labels_list = []

        if epoch == -1:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch

    def keep_features(self, inputs, targets):
        self.feats_list.append(inputs)
        self.labels_list.append(targets)

    def forward(self, inputs, targets, avg_factor=None):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        # Completely bypass computation if zero
        if self.loss_weight == 0 or self.current_epoch < self.loss_warm_up_epochs:
            return 0

        n = inputs.size(0)

        dist = pairwise_euclidean(inputs)

        dmax = dist.max()
        dmin = dist.min()

        if dmax - dmin < 1e-3:
            warn('Network has collapsed, both positive and negative samples are mapped (around) to single point')

        large_num = 10000.0
        masker = torch.ones_like(dist)

        feats = inputs.clone().detach().to(dtype=torch.float16)
        labels = targets.clone().detach()
        self.keep_features(feats, labels)

        if self.trainer is not None and self.current_epoch >= self.population_warm_up_epochs:
            # 1. Find inclusion labels - NxC labels
            inclusion_labels = self.trainer.predict_per_class(inputs.clone().detach())

            max_target_label = targets.max()
            if (max_target_label + 1) > inclusion_labels.shape[1]:
                raise AssertionError(
                    f'Assumption that targets\' labels are 0...C-1, where C is the number of classes, does not hold:'
                    f' target_label: {max_target_label}, # trained openset models: {inclusion_labels.shape[1]}')

            for i in range(n):
                inclusion_vec = inclusion_labels[i]

                # (A)
                # if the sample is not included in its corresponding class, we discard the sample from the possibility
                # to be selected as hard positive by assigning a small distance to all cells of the same class
                if inclusion_vec[int(targets[i])] == 0:
                    # Below we visit the rows that are same label as sample i and i-th column, otherwise the distances
                    # of all same-class samples to the i-th sample
                    masker[targets == targets[i], i] = 0  #dmin
                    for sample_index in torch.where(targets == targets[i]):
                        self.ei.add_hard_positive(sample_class=int(targets[i]), sample_index=sample_index)

                # (B)
                # if the sample is included in a class other than its corresponding class, we discard the sample from
                # the possibility to be selected as hard negative by assigning a large distance to all cells of the
                # other classes

                # create a mask initialized by all True
                selected_indices = torch.ones((inclusion_vec.shape[0],), dtype=torch.bool)
                # exclude same class index
                selected_indices[int(targets[i])] = False
                # check if inclusion label of the sample corresponds to another class and if so, exclude the
                # corresponding samples as described above
                if any(inclusion_vec[selected_indices]):
                    # Below we visit the rows that are not the same label as sample i and i-th column, otherwise the
                    # distances of all different-class samples to the i-th sample
                    masker[targets != targets[i], i] = large_num #dmax
                    for sample_index in torch.where(targets != targets[i]):
                        self.ei.add_hard_negative(sample_class=int(targets[i]), sample_index=sample_index)

            self.ei.advance_iteration()
            dist = dist * masker
        else:
            warn('Openset training not available')

        # Create a rectangular mask that is 1 in indices where class is the same and 0 elsewhere
        # e.g. indices in diagonal ([1,1], [2,2], [3,3]) are always 1 because they correspond to the same sample
        # for the rest e.g. if sample 1 is the same class with sample 2, then [1,2] will be 1, else 0
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # For each anchor, find the hardest positive and negative
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))

            # Semi-hard negative mining
            if self.current_epoch < self.semi_hard_warm_up_epochs:
                negative_indices = mask[i] == 0
                gt_positive_indices = dist_ap[-1] < dist[i]
                #lt_margin_indices = dist[i] < self.margin
                reduced_indices = negative_indices & gt_positive_indices
                n_reduced_indices = torch.count_nonzero(reduced_indices)
                #print(f'All {torch.count_nonzero(negative_indices)}, '
                #      f'Reduced {n_reduced_indices}')
                if n_reduced_indices > 0:
                    dist_an.append(dist[i][reduced_indices].min().unsqueeze(0))
                else:
                    dist_ap.pop()
            else:
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # print('Dists AP NZ:', torch.count_nonzero(dist_ap))

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return self.loss_weight * loss
