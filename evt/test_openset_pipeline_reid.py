from typing import Optional, Sequence
import torch
from pathlib import Path
import utils
from vast_openset import *
from features_storage import FeaturesStorage
from sklearn.decomposition import PCA
from timeit import default_timer as timer


def filter_labels_range(feats: torch.Tensor, labels: torch.Tensor, retained_labels: Optional[Sequence]) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    keep_mask = torch.zeros(labels.shape, dtype=torch.bool)

    for l in retained_labels:
        kept_indices = torch.where(labels == l)
        keep_mask[kept_indices] = True

    return feats[keep_mask], labels[keep_mask]


def filter_labels_populations(feats: torch.Tensor, labels: torch.Tensor, threshold: int = 10):
    keep_mask = torch.zeros(labels.shape, dtype=torch.bool)
    populations, corresponding_labels = utils.count_populations(labels)

    for i, l in enumerate(corresponding_labels):
        if populations[i] >= threshold:
            kept_indices = torch.where(labels == l)
            keep_mask[kept_indices] = True

    return feats[keep_mask], labels[keep_mask]


def filter_labels(feats: torch.Tensor, labels: torch.Tensor, retained_labels: Optional[Sequence], threshold: int = 10):
    return filter_labels_populations(*filter_labels_range(feats, labels, retained_labels), threshold)


def load_data(cached_path, reduce=True, exclude_zero=True, testing_to_dirty=None):
    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    # We entirely omit background images. In query images they do not exist but we do the same for completeness.
    (raw_training_feats, raw_testing_feats), (raw_training_labels, raw_testing_labels) = fs.raw_features()
    raw_training_indices, raw_testing_indices = fs.training_feats['data_idx'], fs.testing_feats['data_idx']

    if testing_to_dirty is not None:
        dirty_indices = torch.load(testing_to_dirty)
        # assume testing indices are in ascending order because no sampling or randomization has been applied
        assert list(raw_testing_indices.numpy()) == list(range(0, len(raw_testing_indices)))
        raw_testing_labels[dirty_indices == 1] = raw_testing_labels.max() + 1

    if exclude_zero:
        raw_training_feats = raw_training_feats[raw_training_labels != 0]
        raw_testing_feats = raw_testing_feats[raw_testing_labels != 0]

        raw_training_indices = raw_training_indices[raw_training_labels != 0]
        raw_testing_indices = raw_testing_indices[raw_testing_labels != 0]

        raw_training_labels = raw_training_labels[raw_training_labels != 0]
        raw_testing_labels = raw_testing_labels[raw_testing_labels != 0]

    # keep = list(range(1, 300))
    # raw_training_feats, raw_training_labels = \
    #     filter_labels(raw_training_feats, raw_training_labels, retained_labels=keep, threshold=20)
    #
    # raw_testing_feats, raw_testing_labels = \
    #     filter_labels(raw_testing_feats, raw_testing_labels, retained_labels=torch.unique(raw_training_labels),
    #                   threshold=0)

    # print(utils.count_populations(raw_training_labels))

    print(f'Experiment with {len(torch.unique(raw_training_labels))} classes')

    if reduce:
        pca = PCA(n_components=512)

        reduced_training_feats = torch.from_numpy(
            utils.measure_time(message='PCA fit_transform', fun=pca.fit_transform, **{'X': raw_training_feats}))
        reduced_testing_feats = torch.from_numpy(
            utils.measure_time(message='PCA transform', fun=pca.transform, **{'X': raw_testing_feats}))
    else:
        reduced_training_feats = raw_training_feats
        reduced_testing_feats = raw_testing_feats

    training_data = OpensetData(reduced_training_feats, raw_training_labels, raw_training_indices, None)
    testing_data = OpensetData(reduced_testing_feats, raw_testing_labels, raw_testing_indices, None)

    return training_data, testing_data

example_configurations = \
    [
        # ('OpenMax', "--distance_metric     euclidean --tailsize 1.0"),
        # ('OpenMax', "--distance_metric euclidean --tailsize 0.5"),
        ('OpenMax', "--distance_metric euclidean --tailsize 0.25"),
        # ('EVM', "--distance_metric euclidean --distance_multiplier 0.7"),
        # ('EVM', "--distance_metric euclidean --tailsize 0.1"),
        # ('MultiModalOpenMax', "--distance_metric euclidean --tailsize 0.25 --Clustering_Algo finch"),
    ]

#embeddings_root = Path('../lightning/features/orig')
embeddings_root = Path('../lightning/pipelines_datasets_scripts/mnist/features/dirtymnist0.1')
assert embeddings_root.exists()
data_root = Path('../lightning/pipelines_datasets_scripts/mnist')
assert data_root.exists()

#cached_path = '/home/amidemo/devel/workspace/object_classifier_deploy/lightning/features/market1501_vanilla_resnet50.pt'

# training_data, testing_data = \
#     load_data(cached_path=embeddings_root / 'market1501_LitPCB_orig_triloss_1.pt', reduce=False)
# training_data_2, testing_data_2 = \
#     load_data(cached_path=embeddings_root / 'market1501_LitPCB_orig_triloss_2.pt', reduce=False)

model_class_name = 'ResNetMNIST'

approach, algorithm_parameters = example_configurations[0]

for epoch_start in range(8, 9):
    epoch_end = epoch_start + 1

    training_data, testing_data = \
        load_data(cached_path=embeddings_root / f'dirtymnist_{model_class_name}_epoch-{epoch_start}.pt', reduce=False,
                  exclude_zero=False, testing_to_dirty=data_root / 'dirty_mnist_test_dp-0.1_limits0.1-0.5-indices.pt')
    training_data_2, testing_data_2 = \
        load_data(cached_path=embeddings_root / f'dirtymnist_{model_class_name}_epoch-{epoch_end}.pt', reduce=False,
                  exclude_zero=False, testing_to_dirty=data_root / 'dirty_mnist_test_dp-0.1_limits0.1-0.5-indices.pt')

    epoch_end = epoch_start + 1
    saver_parameters = f"--OOD_Algo {approach}"
    model_params = OpensetModelParameters(approach, algorithm_parameters, saver_parameters)

    trainer = OpensetTrainer(training_data, model_params, inference_threshold=0.5)

    utils.measure_time(message='Training', fun=trainer.train, **{})

    # eval_args = {'data': testing_data, 'with_noclass': False}
    # utils.measure_time(message='Evaluation', fun=trainer.eval_classification, **eval_args)

    eval_args = {'data': testing_data_2}
    ret = utils.measure_time(message='Evaluation', fun=trainer.eval, **eval_args)
    torch.save(ret[0], f'{epoch_start}-vs-{epoch_end}-included.pt')
    torch.save(ret[1], f'{epoch_start}-vs-{epoch_end}-indices.pt')
    pass
