import os
from typing import Callable
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt

from features_storage import FeaturesStorage
from visualizations.embeddings_visualization import EmbeddingsVisualizer


def multidim_heatmap(features: np.ndarray, probability_function: Callable, n_samples: int,
                         dimensionality_reduction_mapper: UMAP):
    """
    Computes a heatmap given a set of ND features, a function that maps the features into probability and a mapper that
    projects the features into 2 dimensions for visualization purposes.

    @param features: Input features
    @param probability_function: probability function which maps features to probability
    @param n_samples: How many samples to use for heatmap generation
    @param dimensionality_reduction_mapper: the mapper object which projects ND features to 2 dimensions
    @return:
    """

    minimums = np.min(features, axis=0)
    maximums = np.max(features, axis=0)

    feat_dim = features.shape[1]

    # Generate random samples within the min/max range of each dimension
    random_samples = np.random.uniform(low=minimums, high=maximums, size=(n_samples, feat_dim))

    # Project the samples into 2D
    projected_random_samples = dimensionality_reduction_mapper.transform(random_samples)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax_bound = EmbeddingsVisualizer.axis_bounds(np.vstack(projected_random_samples))

    scatter = ax.scatter(*projected_random_samples.T, s=2, alpha=0.3)
    ax.axis(ax_bound)
    ax.set(xticks=[], yticks=[])
    #ax.set_title(f'{features_id} - {part}')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # 1. Select dataset features, FileStorage output format, convert them in OpensetData format
    features_path = '../lightning/features/MNIST_LitModel.pt'
    assert os.path.exists(features_path)

    fs = FeaturesStorage(cached_path=features_path)

    ev = EmbeddingsVisualizer('.')
    ev.add_features('MNIST_LitModel', fs)
    feats = fs['feats']['test']['feats'].numpy()
    multidim_heatmap(feats, lambda dummy: None, 100000, ev.get_mapper('MNIST_LitModel'))

    pass