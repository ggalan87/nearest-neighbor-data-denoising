import argparse

import torch
import pickle
from pathlib import Path
from itertools import product
from features_storage import FeaturesStorage
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from lightning_lite.utilities.seed import seed_everything
from lightning.data.dataset_utils import random_split_perc

from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, MulticlassConfusionMatrix, BinaryF1Score


def load_data(cached_path, reduce=True, exclude_zero=True, testing_to_dirty=None):
    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    # We entirely omit background images. In query images they do not exist but we do the same for completeness.
    (raw_training_feats, raw_testing_feats), (raw_training_labels, raw_testing_labels) = fs.raw_features()
    # raw_training_indices, raw_testing_indices = fs.training_feats['data_idx'], fs.testing_feats['data_idx']

    #raw_testing_feats = raw_testing_feats[(raw_testing_labels != 8) & (raw_testing_labels != 9)]
    #raw_testing_labels = raw_testing_labels[(raw_testing_labels != 8) & (raw_testing_labels != 9)]
    return raw_training_feats, raw_training_labels, raw_testing_feats, raw_testing_labels


def run(args):
    dataset_name = args.dataset_name
    target_dataset_name = args.target_dataset_name
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

    test_only_acc = args.test_only_accuracy
    self_test = args.self_test

    gallery_indices, query_indices = None, None

    seed_everything(13)

    # Override the option above, as self test is done on full test set on its own
    if self_test:
        test_only_acc = True

    for v in versions:
        print(f'Computing accuracies for version {v}...')
        run_path = \
            Path(
                f'./lightning_logs/{dm_name}_{model_class_name}/version_{v}')

        version_accuracies = []

        features_folder_name = 'features' if target_dataset_name is None else f'features-{target_dataset_name.lower()}'
        for e in epochs:
            cached_path = run_path / features_folder_name / f'features_epoch-{e}.pt'

            if not cached_path.exists():
                continue

            #print(cached_path)
            train_embeddings, train_labels, test_embeddings, test_labels = load_data(cached_path)

            if args.batched_knn:
                distance_fn = LpDistance(normalize_embeddings=False, power=2)
                custom_knn = CustomKNN(distance_fn, batch_size=2048)
                accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1, knn_func=custom_knn)
            else:
                accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

            if test_only_acc:
                if self_test:
                    accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels)
                else:
                    if gallery_indices is None or query_indices is None:
                        print('Generating indices')
                        gallery_indices, query_indices = random_split_perc(len(test_embeddings), 0.5)

                    accuracies = accuracy_calculator.get_accuracy(
                        test_embeddings[query_indices], test_labels[query_indices], test_embeddings[gallery_indices],
                        test_labels[gallery_indices]
                    )
            else:
                accuracies = accuracy_calculator.get_accuracy(
                    test_embeddings, test_labels, train_embeddings, train_labels
                )

                # multiclass_confusion_matrix_metric = \
                #     MulticlassConfusionMatrix(num_classes=torch.unique(train_labels), normalize='true')
                #
                # predicted_class_labels =
                # multiclass_confusion_matrix_metric(predicted_class_labels, test_labels)

            version_accuracies.append(accuracies["precision_at_1"])

        acc_filename = 'acc.pkl' if target_dataset_name is None else f'acc-{target_dataset_name.lower()}.pkl'
        with open(run_path / acc_filename, 'wb') as f:
            pickle.dump(version_accuracies, f)
        print(version_accuracies)


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-name', required=True, type=str,
                        help='The name of the dataset, or more precisely the class name.')
    parser.add_argument('--target-dataset-name', required=False, type=str,
                        help='The name of the dataset from which to extract features, or more precisely the class name.'
                             'If not specified it will be the same as dataset-name arg.')
    parser.add_argument('--model-name', required=True, type=str,
                        help='The name of the model, or more precisely the model class name.')

    versions_group = parser.add_mutually_exclusive_group(required=True)
    versions_group.add_argument('--versions-list', nargs='+', type=int, help='List of versions')
    versions_group.add_argument('--versions-range', nargs=2, type=int, help='Range of versions (min,max included)')

    epochs_group = parser.add_mutually_exclusive_group(required=True)
    epochs_group.add_argument('--epochs-list', nargs='+', type=int, help='List of epochs')
    epochs_group.add_argument('--epochs-range', nargs=2, type=int, help='Range of epochs (min,max included)')

    parser.add_argument('--test-only-accuracy', required=True, type=bool,
                        help='Whether to use only test data for computing accuracy, and split them or use self test')
    parser.add_argument('--self-test', required=True, type=bool,
                        help='Whether to use the same data for gallery and query sets.')
    parser.add_argument('--batched-knn', required=True, type=bool,
                        help='Whether to use the custom batched knn. Useful for large datasets where FAISS results to'
                             'memory issues.')

    return parser.parse_args()


if __name__ == '__main__':
    run(parse_cli())

