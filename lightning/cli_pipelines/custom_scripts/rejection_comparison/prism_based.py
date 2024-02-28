import torch
from torchmetrics.classification import BinaryConfusionMatrix, BinaryF1Score

from common import load_features, to_list
from pytorch_metric_learning.losses import CrossBatchMemory
from pytorch_metric_learning.losses import ContrastiveLoss

from torch_metric_learning.miners.prism import PRISM

from inspect_openset_model_reduction import MetricsOverTimeLogger


def run(dataset_name, model_name, version, reference_epoch, batch_size, samples_fraction=1.0):
    reference_data, training_data = load_features(dataset_name, model_name, version, reference_epoch)

    reference_feats, reference_labels, reference_indices, reference_is_noisy = reference_data
    feats, labels, indices, is_noisy = training_data

    xbm_object = CrossBatchMemory(ContrastiveLoss(), embedding_size=reference_feats.shape[1],
                                  memory_size=reference_feats.shape[0], miner=None)

    miner = PRISM(num_classes=len(torch.unique(reference_labels)), cross_batch_memory_object=xbm_object)

    n_samples = len(reference_labels)
    reference_feats_l = to_list(reference_feats, batch_size=batch_size)
    reference_labels_l = to_list(reference_labels, batch_size=batch_size)

    for batch_feats, batch_labels in zip(reference_feats_l, reference_labels_l):
        xbm_object(batch_feats, batch_labels)

    miner._last_clean_labels = torch.unique(reference_labels)
    miner._update_centroids()

    n_samples = len(labels)
    feats_l = to_list(feats, batch_size=batch_size)
    labels_l = to_list(labels, batch_size=batch_size)
    indices_l = to_list(indices, batch_size=batch_size)
    isnoisy_l = to_list(is_noisy, batch_size=batch_size)

    confusion_matrix_metric = BinaryConfusionMatrix(normalize='true')
    f1_score_metric = BinaryF1Score()

    all_noisy_predictions = []

    for idx, (batch_feats, batch_labels, batch_indices, batch_is_noisy) in \
            enumerate(zip(feats_l, labels_l, indices_l, isnoisy_l)):

        miner(batch_feats.cuda(), batch_labels.cuda())

        predicted_as_clean = miner._clean_in_batch
        gt_clean = torch.logical_not(batch_is_noisy)

        all_noisy_predictions.extend(torch.logical_not(predicted_as_clean))

        # confusion_matrix = confusion_matrix_metric(predicted_as_clean, gt_clean)
        # print(confusion_matrix)

    all_predicted_as_clean = torch.logical_not(torch.BoolTensor(all_noisy_predictions))
    all_gt_clean = torch.logical_not(is_noisy)

    confusion_matrix = confusion_matrix_metric(all_predicted_as_clean, all_gt_clean)
    f1_score = f1_score_metric(all_predicted_as_clean, all_gt_clean)

    stats = {'confusion_matrix': confusion_matrix, 'f1_score': f1_score}
    return stats
