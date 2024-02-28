import random
import numpy as np
from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10


class DatasetVisualizer:
    def __init__(self, n_per_class):
        self._n_per_class = n_per_class

    def visualize(self, dataset: VisionDataset):
        unique_targets = np.unique(dataset.targets)
        for t in unique_targets:
            t_data = dataset.data[dataset.targets == t]
            indices = random.sample(range(len(t_data)), self._n_per_class)
            for idx in indices:
                plt.figure()
                plt.imshow(t_data[idx])
                plt.title(dataset.classes[t])
        plt.show()


if __name__ == '__main__':
    cifar_dataset = CIFAR10('../lightning/data', train=False)
    viz = DatasetVisualizer(n_per_class=1)
    viz.visualize(cifar_dataset)
