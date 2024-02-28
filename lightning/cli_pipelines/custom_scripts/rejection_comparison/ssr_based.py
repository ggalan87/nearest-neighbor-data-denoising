import torch
from torchmetrics.classification import BinaryConfusionMatrix

from common import load_features, to_list
from pytorch_metric_learning.losses import CrossBatchMemory
from pytorch_metric_learning.losses import ContrastiveLoss
from ssr_utils import weighted_knn
from nn_torch import NN


def run(dataset_name, model_name, version, reference_epoch, batch_size):
    reference_data, training_data = load_features(dataset_name, model_name, version, reference_epoch)

    reference_feats, reference_labels, reference_indices, reference_is_noisy = reference_data
    feats, labels, indices, is_noisy = training_data

    feature_bank = reference_feats

    # We don't have a linear classifier so we utilize an NN
    nn_cls = NN(X=reference_feats.cuda(), Y=reference_labels.cuda())

    ref_feats_l = to_list(reference_feats.cuda())

    prediction = []
    for ref_feats_i in ref_feats_l:
        y_pred = nn_cls.predict(ref_feats_i)
        prediction.append(y_pred)

    # Unfortunately we don't have the notion of score !! Maybe I need distance threshold?

    ################################### sample relabelling ###################################
    # 1. Pass classifier scores from softmax
    # prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
    # 2. Predict class with maximum score
    # his_score, his_label = prediction_cls.max(1)
    # print(f'Prediction track: mean: {his_score.mean()} max: {his_score.max()} min: {his_score.min()}')
    # 3. Get indices where scores surpass theta_r threshold
    # conf_id = torch.where(his_score > args.theta_r)[0]
    # 4. Modify labels at indices where the score is high
    # modified_label = torch.clone(noisy_label).detach()
    # modified_label[conf_id] = his_label[conf_id]


    ################################### sample selection ###################################
    # prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k,
    #                               10)  # temperature in weighted KNN
    # The ratio of the label
    # vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
    # vote_max = prediction_knn.max(dim=1)[0]
    # right_score = vote_y / vote_max
    # clean_id = torch.where(right_score >= args.theta_s)[0]
    # noisy_id = torch.where(right_score < args.theta_s)[0]
