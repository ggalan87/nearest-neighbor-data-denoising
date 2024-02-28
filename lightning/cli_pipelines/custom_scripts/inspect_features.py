import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN
from features_storage import FeaturesStorage


def load_data(cached_path, reduce=True, exclude_zero=True, testing_to_dirty=None):
    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    # We entirely omit background images. In query images they do not exist but we do the same for completeness.
    (raw_training_feats, raw_testing_feats), (raw_training_labels, raw_testing_labels) = fs.raw_features()
    # raw_training_indices, raw_testing_indices = fs.training_feats['data_idx'], fs.testing_feats['data_idx']

    #raw_testing_feats = raw_testing_feats[(raw_testing_labels != 8) & (raw_testing_labels != 9)]
    #raw_testing_labels = raw_testing_labels[(raw_testing_labels != 8) & (raw_testing_labels != 9)]
    return raw_training_feats, raw_training_labels, raw_testing_feats, raw_testing_labels


feats_path = '/home/amidemo/devel/workspace/object_classifier_deploy/lightning/cli_pipelines/lightning_logs/food101n_LitInception/version_0/features/features_epoch-0.pt'

_, _, testing_feats, testing_labels = load_data(feats_path)

distance_fn = LpDistance(normalize_embeddings=False, power=2)
custom_knn = CustomKNN(distance_fn, batch_size=2048)
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1, knn_func=custom_knn)
accuracy = accuracy_calculator.get_accuracy(testing_feats, testing_labels)
print(accuracy)
