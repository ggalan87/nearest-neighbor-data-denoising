import argparse

import torch
import pickle
from pathlib import Path
from itertools import product
from features_storage import FeaturesStorage
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from lightning_lite.utilities.seed import seed_everything
from lightning.data.dataset_utils import random_split_perc

from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, MulticlassConfusionMatrix, BinaryF1Score


def load_data(cached_path, reduce=True, exclude_zero=True, testing_to_dirty=None):
    fs = FeaturesStorage(cached_path=cached_path, target_key='target_orig')

    # We entirely omit background images. In query images they do not exist but we do the same for completeness.
    (raw_training_feats, raw_testing_feats), (raw_training_labels, raw_testing_labels) = fs.raw_features()
    # raw_training_indices, raw_testing_indices = fs.training_feats['data_idx'], fs.testing_feats['data_idx']

    #raw_testing_feats = raw_testing_feats[(raw_testing_labels != 8) & (raw_testing_labels != 9)]
    #raw_testing_labels = raw_testing_labels[(raw_testing_labels != 8) & (raw_testing_labels != 9)]
    return raw_training_feats, raw_training_labels, raw_testing_feats, raw_testing_labels


def run(args):
    dataset_name = args.dataset_name
    model_class_name = args.model_name
    dm_name = dataset_name.lower()

    if args.versions_range is not None:
        versions = list(range(args.versions_range[0], args.versions_range[1] + 1))
    else:
        versions = args.versions_list

    if args.epochs_range is not None:
        epochs = list(range(args.epochs_range[0], args.epochs_range[1] + 1))
    else:
        epochs = args.epochs_list

    seed_everything(13)

    for v in versions:
        print(f'Computing accuracies for version {v}...')
        run_path = \
            Path(
                f'../lightning_logs/{dm_name}_{model_class_name}/version_{v}')

        version_train_self_accuracies = []
        version_test_self_accuracies = []

        for e in epochs:
            cached_path = run_path / 'features' / f'features_epoch-{e}.pt'

            if not cached_path.exists():
                continue

            #print(cached_path)
            train_embeddings, train_labels, test_embeddings, test_labels = load_data(cached_path)
            accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

            train_self_accuracies = accuracy_calculator.get_accuracy(train_embeddings, train_labels)
            test_self_accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels)

            version_train_self_accuracies.append(train_self_accuracies["precision_at_1"])
            version_test_self_accuracies.append(test_self_accuracies["precision_at_1"])

        print(version_train_self_accuracies)
        print(version_test_self_accuracies)


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-name', required=True, type=str,
                        help='The name of the dataset, or more precisely the class name.')
    parser.add_argument('--model-name', required=True, type=str,
                        help='The name of the model, or more precisely the model class name.')

    versions_group = parser.add_mutually_exclusive_group(required=True)
    versions_group.add_argument('--versions-list', nargs='+', type=int, help='List of versions')
    versions_group.add_argument('--versions-range', nargs=2, type=int, help='Range of versions (min,max included)')

    epochs_group = parser.add_mutually_exclusive_group(required=True)
    epochs_group.add_argument('--epochs-list', nargs='+', type=int, help='List of epochs')
    epochs_group.add_argument('--epochs-range', nargs=2, type=int, help='Range of epochs (min,max included)')

    return parser.parse_args()


if __name__ == '__main__':
    run(parse_cli())

