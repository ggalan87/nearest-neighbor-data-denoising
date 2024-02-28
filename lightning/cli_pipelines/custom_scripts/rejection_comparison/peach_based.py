import torch
from torch_metric_learning.miners.openset_rejection.rejection_strategies import ClusterAwarePairRejection
from common import load_features, to_list
from torchmetrics.classification import BinaryConfusionMatrix


def run(dataset_name, model_name, version, reference_epoch, batch_size):
    reference_data, training_data = load_features(dataset_name, model_name, version, reference_epoch)

    reference_feats, reference_labels, reference_indices, reference_is_noisy = reference_data
    feats, labels, indices, is_noisy = training_data

    rejection_strategy = ClusterAwarePairRejection()

    rejection_strategy.keep_epoch_features(reference_feats, reference_labels)

    rejection_strategy.train()

    n_samples = len(labels)
    feats_l = to_list(feats, n_samples=n_samples, batch_size=batch_size)
    labels_l = to_list(labels, n_samples=n_samples, batch_size=batch_size)
    indices_l = to_list(indices, n_samples=n_samples, batch_size=batch_size)
    isnoisy_l = to_list(is_noisy, n_samples=n_samples, batch_size=batch_size)

    confusion_matrix_metric = BinaryConfusionMatrix(normalize='true')

    for idx, (batch_feats, batch_labels, batch_indices, batch_is_noisy) in \
            enumerate(zip(feats_l, labels_l, indices_l, isnoisy_l)):

        if idx != 0:
            continue

        rejection_strategy.predict_noise(batch_feats.cuda(), batch_labels.cuda())
        inclusion_probabilities = rejection_strategy.retrieve_batch_predictions()

        predicted_as_clean = []
        for i, sample_probs in enumerate(inclusion_probabilities):
            if sample_probs[batch_labels[i]] == 1.0:
                predicted_as_clean.append(True)
            else:
                predicted_as_clean.append(False)

        predicted_as_clean = torch.tensor(predicted_as_clean)
        gt_clean = torch.logical_not(batch_is_noisy)

        confusion_matrix = confusion_matrix_metric(predicted_as_clean, gt_clean)
        print(confusion_matrix)
