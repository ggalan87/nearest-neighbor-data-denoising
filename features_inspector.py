import torch
import pandas as pd
import miniball


class FeaturesInspector:
    """
    A class which performs inspections on features sets
    """
    def __init__(self):
        self.separator = '*' * 50

    def process(self, part: str, features: torch.Tensor, labels: torch.Tensor):
        if len(features) != len(labels):
            raise AssertionError(f'Input tensors dimensions do not match {len(features)} vs {len(labels)}')

        # Process individual inspection methods
        #self.distances_stats(part, features, labels)
        self.minimum_bounding_hypersphere(part, features, labels)

    def distances_stats(self, part: str, features: torch.Tensor, labels: torch.Tensor):
        """
        Computes statistics based on distances, both intra-class and inter-class

        @param part:
        @param features:
        @param labels:
        @return:
        """
        n_samples = len(features)
        mask = labels.expand(n_samples, n_samples).eq(labels.expand(n_samples, n_samples).t())
        dist_mat = torch.cdist(features, features)
        intra_class_distances = dist_mat[mask]
        inter_class_distances = dist_mat[mask == 0]

        intra_std, intra_mean = torch.std_mean(intra_class_distances)
        inter_std, inter_mean = torch.std_mean(inter_class_distances)

        print(f'Global features stats for part {part}:\n'
              f'Intra-class distances -> std:{intra_std} mean: {intra_mean} norm_std: {intra_std / intra_mean}\n'
              f'Inter-class distances -> std:{inter_std} mean: {inter_mean} norm_std: {inter_std / inter_mean}'
              f'{self.separator}')

        print(f'Per class features stats for part {part}:')

        for l in torch.unique(labels):
            l_positive = dist_mat[labels == l][mask[labels == l]]
            l_negative = dist_mat[labels == l][mask[labels == l] == 0]
            l_std, l_mean = torch.std_mean(l_positive)
            nl_std, nl_mean = torch.std_mean(l_negative)
            print(f'Intra-class distances of {l} -> std:{l_std} mean: {l_mean} norm_std: {l_std / l_mean}\n'
                  f'Inter-class distances of {l} -> std:{nl_std} mean: {nl_mean} norm_std: {nl_std / nl_mean}')

        print(f'{self.separator}')

    def minimum_bounding_hypersphere(self, part: str, features: torch.Tensor, labels: torch.Tensor):
        """
        Computes minimum enclosing ball (MEB) on points corresponding to each class, for comparison about compactness
        of the points per class

        TODO: Utilize a method with outliers for more robust results
        https://github.com/tomholmes19/Minimum-Enclosing-Balls-with-Outliers
        The above requires commercial optimizers, refactor to nlopt(?) - https://github.com/stevengj/nlopt

        @param part:
        @param features:
        @param labels:
        @return:
        """
        print(f'Per class bounding circle for part {part}:')

        for l in torch.unique(labels):
            l_features = features[labels == l]
            _, r = miniball.get_bounding_ball(l_features.numpy())
            print(f'Minimum bounding hypersphere for label {l}: {r}')
        print(f'{self.separator}')

