from typing import Optional, Sequence
import torch
import torch.nn.functional as F
from pathlib import Path
import utils
from vast_openset import *
from features_storage import FeaturesStorage
from sklearn.decomposition import PCA
from timeit import default_timer as timer


def load_data(gallery_path, query_path, reduce=True, exclude_zero=True, testing_to_dirty=None):
    fs_gallery = FeaturesStorage(cached_path=gallery_path, target_key='target')
    fs_query = FeaturesStorage(cached_path=query_path, target_key='target')

    # Choose training data from both epochs (they have noisy labels)
    (gallery_feats, _), (gallery_labels, _) = fs_gallery.raw_features()
    (query_feats, _), (query_labels, _) = fs_query.raw_features()
    gallery_indices = fs_gallery.training_feats['data_idx']
    query_indices = fs_query.training_feats['data_idx']

    # Assign noisy data from testing to another class. This is for measuring openset accuracy having ground-truth labels
    query_noisy = fs_query.training_feats['is_noisy']
    query_labels[query_noisy] = query_labels.max() + 1

    gallery_noisy = fs_query.training_feats['is_noisy']
    # gallery_feats = gallery_feats[gallery_noisy == False]
    # gallery_labels = gallery_labels[gallery_noisy == False]
    # gallery_indices = gallery_indices[gallery_noisy == False]

    random_indices = \
        torch.randperm(len(gallery_labels))[:int(0.3 * len(gallery_labels))]
    gallery_feats = gallery_feats[random_indices]
    gallery_labels = gallery_labels[random_indices]
    gallery_indices = gallery_indices[random_indices]

    gallery_feats = F.normalize(gallery_feats)
    query_feats = F.normalize(query_feats)

    training_data = OpensetData(gallery_feats, gallery_labels, gallery_indices, None)
    testing_data = OpensetData(query_feats, query_labels, query_indices, None)

    return training_data, testing_data


epoch_from = 1
epoch_to = 2
dataset_name = 'NoisyMNISTSubset'
dm_name = dataset_name.lower()
model_class_name = 'LitModel'
version = 0
epochs = list(range(10))

run_path = Path(
    f'../lightning/pipelines_datasets_scripts/mnist/lightning_logs/{dm_name}_{model_class_name}/version_{version}')
gallery_path = run_path / 'features' / f'features_epoch-{epoch_from}.pt'
query_path = run_path / 'features' / f'features_epoch-{epoch_to}.pt'
#print(cached_path)

training_data, testing_data = load_data(gallery_path, query_path)
example_configurations = \
    [
        # ('OpenMax', "--distance_metric     euclidean --tailsize 1.0"),
        ('OpenMax', "--distance_metric euclidean --tailsize 0.5"),
        # ('OpenMax', "--distance_metric euclidean --tailsize 0.25"),
        # ('EVM', "--distance_metric euclidean --distance_multiplier 1.0"),
        # ('EVM', "--distance_metric euclidean --tailsize 0.7"),
        # ('MultiModalOpenMax', "--distance_metric euclidean --tailsize 0.25 --Clustering_Algo finch"),
    ]

approach, algorithm_parameters = example_configurations[0]

saver_parameters = f"--OOD_Algo {approach}"

model_params = OpensetModelParameters(approach, algorithm_parameters, saver_parameters)

trainer = OpensetTrainer(training_data, model_params, inference_threshold=0.5)

utils.measure_time(message='Training', fun=trainer.train, **{})

eval_args = {'data': testing_data}
ret = utils.measure_time(message='Evaluation', fun=trainer.eval, **eval_args)

# eval_args = {'data': testing_data, 'with_noclass': False}
# utils.measure_time(message='Evaluation', fun=trainer.eval_classification, **eval_args)