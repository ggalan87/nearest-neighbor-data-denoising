from pathlib import Path

from sklearn.decomposition import PCA

from features_storage import FeaturesStorage
from evt.vast_openset import *
from torch.nn import functional as F

from pytorch_lightning.utilities.seed import seed_everything


def to_list(t):
    t_l = []
    for i in range(125):
        t_l.append(t[i * 64:64 * (i + 1)])

    t_l.append(t[8000:8032])
    return t_l


# def load_data(cached_path, samples_fraction=0.3, reduced_dim=-1):
#     fs = FeaturesStorage(cached_path=cached_path, target_key='target')
#
#     # We entirely omit background images. In query images they do not exist but we do the same for completeness.
#     (raw_training_feats, _), (raw_training_labels, _) = fs.raw_features()
#     raw_training_indices = fs.training_feats['data_idx']
#     raw_training_is_noisy = fs.training_feats['is_noisy']
#
#     if samples_fraction < 1.0:
#         random_indices = torch.randperm(len(raw_training_labels))[:int(samples_fraction * len(raw_training_labels))]
#     else:
#         random_indices = torch.arange(len(raw_training_labels))
#
#     training_feats = raw_training_feats[random_indices]
#
#     if reduced_dim != -1:
#         pca = PCA(n_components=reduced_dim)
#         training_feats = torch.from_numpy(pca.fit_transform(X=training_feats))
#     else:
#         pca = None
#
#     return training_feats, raw_training_labels[random_indices], raw_training_indices[random_indices], \
#         raw_training_is_noisy[random_indices], pca
#
#
# def simulate_noisy_rejection(dataset_name, model_name, version, reference_epoch, batch_size, openset_options):
#     dataset_name = dataset_name
#     model_class_name = model_name
#     dm_name = dataset_name.lower()
#
#     training_epoch = reference_epoch + 1
#
#     run_path = \
#         Path(
#             f'../lightning_logs/{dm_name}_{model_class_name}/version_{version}')
#
#     cached_path_reference = run_path / 'features' / f'features_epoch-{reference_epoch}.pt'
#     cached_path_training = run_path / 'features' / f'features_epoch-{training_epoch}.pt'
#
#     approach = openset_options['approach']
#     distance_metric = openset_options['distance_metric']
#     tail_size = openset_options['tail_size']
#
#     reference_feats, reference_labels, reference_indices, reference_is_noisy, pca = \
#         load_data(cached_path_reference, samples_fraction=0.3, reduced_dim=openset_options['reduced_dim'])
#
#     algorithm_parameters_string = f'--distance_metric {distance_metric} --tailsize {tail_size}'
#
#     rejection_strategy =\
#         PopulationAwarePairRejection(approach=approach, algorithm_parameters_string=algorithm_parameters_string)
#
#     rejection_strategy.keep_epoch_features(reference_feats, reference_labels)
#     rejection_strategy.train()
#
#     feats, labels, indices, is_noisy, _ = \
#         load_data(cached_path_training, samples_fraction=1.0, reduced_dim=-1)
#
#     #
#     feats = to_list(feats)
#     labels = to_list(labels)
#     indices = to_list(indices)
#     is_noisy = to_list(is_noisy)
#
#     for idx, (batch_feats, batch_labels, batch_indices, batch_is_noisy) in \
#             enumerate(zip(feats, labels, indices, is_noisy)):
#
#         # Inclusion probabilities - NxC
#
#         if idx != 0:
#             continue
#
#         batch_feats = F.normalize(batch_feats)
#
#         if pca is not None:
#             batch_feats = torch.from_numpy(pca.transform(X=batch_feats))
#
#         # rejection_strategy.store_batch_predictions(batch_feats, batch_labels)
#         # batch_predictions = rejection_strategy.retrieve_batch_predictions()
#
#         batch_labels[batch_is_noisy == True] = len(rejection_strategy.labels_to_indices) + 1
#         _, _, metrics = rejection_strategy._openset_trainer.eval(OpensetData(batch_feats, batch_labels, batch_indices, None),
#                                              get_metrics=True)
#
#         confusion_matrix = metrics['confusion']
#
#         if confusion_matrix[0, 0] > confusion_matrix[0, 1] and confusion_matrix[1, 1] > confusion_matrix[1, 0]:
#             print(confusion_matrix)
#             return True
#
#         return False
#
#         # Normalize them such that they sum up to 1 per sample
#         # batch_predictions = torch.nn.functional.normalize(batch_predictions, p=1.0)
#
#         # clean_weights = batch_predictions[batch_is_noisy == False]
#         # noisy_weights = batch_predictions[batch_is_noisy == True]
#
#         # print(torch.mean(torch.median(clean_weights, dim=1)),
#         #       torch.mean(torch.median(noisy_weights, dim=1)))
#
#         # clean_weights = []
#         # noisy_weights = []
#
#         # for i in range(len(batch_labels)):
#         #     preds_i = batch_predictions[i]
#         #     preds_i = preds_i[preds_i > 0.5]
#         #     # print(bool(batch_is_noisy[i]), float(torch.mean(preds_i)))
#         #
#         #     if batch_is_noisy[i]:
#         #         noisy_weights.append(torch.mean(preds_i))
#         #     else:
#         #         clean_weights.append(torch.mean(preds_i))
#
#         # print(torch.mean(torch.tensor(clean_weights)) > torch.mean(torch.tensor(noisy_weights)))
#
#         # if idx == 0:
#         #     plt.figure()
#         #     plt.matshow(batch_predictions[batch_is_noisy == False].numpy(), vmin=0, vmax=1)
#         #     plt.colorbar()
#         #     plt.title('noisy samples')
#         #     plt.show()
#         #
#         #     plt.figure()
#         #     plt.matshow(batch_predictions[batch_is_noisy == True].numpy(), vmin=0, vmax=1)
#         #     plt.colorbar()
#         #     plt.title('clean samples')
#         #
#         #     plt.show()
#         #     #plt.savefig(f'{output_dir}/epoch-{epoch}-batch-{idx}.png')


def load_open_data(cached_path, samples_fraction=0.3, reduced_dim=-1):
    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    # We entirely omit background images. In query images they do not exist but we do the same for completeness.
    (raw_training_feats, _), (raw_training_labels, _) = fs.raw_features()
    raw_training_indices = fs.training_feats['data_idx']
    raw_training_is_noisy = fs.training_feats['is_noisy']

    random_indices = \
        torch.randperm(len(raw_training_labels))[:int(samples_fraction * len(raw_training_labels))]
    training_feats = F.normalize(raw_training_feats[random_indices])

    if reduced_dim != -1:
        pca = PCA(n_components=reduced_dim, random_state=13)

        training_feats = torch.from_numpy(pca.fit_transform(X=training_feats))
    else:
        pca = None

    training_data = OpensetData(training_feats, raw_training_labels[random_indices],
                                raw_training_indices[random_indices], None)

    return training_data, raw_training_is_noisy[random_indices], pca


def load_data(cached_path):
    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    (raw_training_feats, _), (raw_training_labels, _) = fs.raw_features()
    raw_training_indices = fs.training_feats['data_idx']
    raw_training_is_noisy = fs.training_feats['is_noisy']

    return F.normalize(raw_training_feats), raw_training_labels, raw_training_indices, raw_training_is_noisy


def simulate_noisy_rejection(dataset_name, model_name, version, reference_epoch, batch_size, openset_options):
    dataset_name = dataset_name
    model_class_name = model_name
    dm_name = dataset_name.lower()

    training_epoch = reference_epoch + 1

    run_path = \
        Path(
            f'../lightning_logs/{dm_name}_{model_class_name}/version_{version}')

    cached_path_reference = run_path / 'features' / f'features_epoch-{reference_epoch}.pt'
    cached_path_training = run_path / 'features' / f'features_epoch-{training_epoch}.pt'

    approach = openset_options['approach']
    distance_metric = openset_options['distance_metric']
    tail_size = openset_options['tail_size']

    data_reference, is_noisy_reference, pca = load_open_data(cached_path_reference,
                                                             reduced_dim=openset_options['reduced_dim'])

    algorithm_parameters_string = f'--distance_metric {distance_metric} --tailsize {tail_size}'
    saver_parameters = f"--OOD_Algo {approach}"
    openset_model_params = OpensetModelParameters(approach, algorithm_parameters_string, saver_parameters)
    openset_trainer = OpensetTrainer(data_reference, openset_model_params, inference_threshold=0.5)

    # print('Openset training')
    openset_trainer.train()

    feats, labels, indices, is_noisy = load_data(cached_path_training)

    #
    feats = to_list(feats)
    labels = to_list(labels)
    indices = to_list(indices)
    is_noisy = to_list(is_noisy)

    for idx, (batch_feats, batch_labels, batch_indices, batch_is_noisy) in \
            enumerate(zip(feats, labels, indices, is_noisy)):

        # Inclusion probabilities - NxC

        if idx != 0:
            continue

        batch_feats = F.normalize(batch_feats)

        print(torch.mean(batch_feats))
        if pca is not None:
            batch_feats = torch.from_numpy(pca.transform(X=batch_feats))

        print(torch.mean(batch_feats))

        batch_predictions = openset_trainer.get_inclusion_probabilities(batch_feats)

        batch_labels[batch_is_noisy == True] = len(openset_trainer.data.labels_to_indices) + 1
        _, _, metrics = openset_trainer.eval(OpensetData(batch_feats, batch_labels, batch_indices, None),
                                             get_metrics=True)

        confusion_matrix = metrics['confusion']

        if confusion_matrix[0, 0] > confusion_matrix[0, 1] and confusion_matrix[1, 1] > confusion_matrix[1, 0]:
            print(confusion_matrix)
        else:
            print(confusion_matrix)
            return False

        # Normalize them such that they sum up to 1 per sample
        # batch_predictions = torch.nn.functional.normalize(batch_predictions, p=1.0)

        clean_weights = batch_predictions[batch_is_noisy == False]
        noisy_weights = batch_predictions[batch_is_noisy == True]

        # print(torch.mean(torch.median(clean_weights, dim=1)),
        #       torch.mean(torch.median(noisy_weights, dim=1)))

        clean_weights = []
        noisy_weights = []

        for i in range(len(batch_labels)):
            preds_i = batch_predictions[i]
            preds_i = preds_i[preds_i > 0.5]
            # print(bool(batch_is_noisy[i]), float(torch.mean(preds_i)))

            if batch_is_noisy[i]:
                noisy_weights.append(torch.mean(preds_i))
            else:
                clean_weights.append(torch.mean(preds_i))

        # print(torch.mean(torch.tensor(clean_weights)) > torch.mean(torch.tensor(noisy_weights)))

        # if idx == 0:
        #     plt.figure()
        #     plt.matshow(batch_predictions[batch_is_noisy == False].numpy(), vmin=0, vmax=1)
        #     plt.colorbar()
        #     plt.title('noisy samples')
        #     plt.show()
        #
        #     plt.figure()
        #     plt.matshow(batch_predictions[batch_is_noisy == True].numpy(), vmin=0, vmax=1)
        #     plt.colorbar()
        #     plt.title('clean samples')
        #
        #     plt.show()
        #     # plt.savefig(f'{output_dir}/epoch-{epoch}-batch-{idx}.png')

    return True


tail_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
reduced_dims = [-1, 32, 64, 128, 256]
approaches = ['OpenMax', 'EVM', 'MultiModalOpenMax']
distance_metrics = ['euclidean', 'cosine']

options = [
    (0.1, 32, 'EVM', 'cosine'), (0.4, 128, 'EVM', 'cosine'), (0.7, 128, 'EVM', 'cosine'),
           (1.0, -1, 'EVM', 'cosine'),
           (1.0, 32, 'EVM', 'euclidean'), (0.1, 32, 'EVM', 'cosine'), (0.4, 128, 'EVM', 'cosine'),
           (0.7, 128, 'EVM', 'cosine'),
           (1.0, -1, 'EVM', 'cosine'), (1.0, 32, 'EVM', 'euclidean'), (0.1, 32, 'EVM', 'cosine'),
           (0.4, 128, 'EVM', 'cosine'), (
               0.7, 128, 'EVM', 'cosine'), (1.0, -1, 'EVM', 'cosine'), (1.0, 32, 'EVM', 'euclidean'),
           (0.1, 32, 'EVM', 'cosine'), (
               0.4, 128, 'EVM', 'cosine'), (0.7, 128, 'EVM', 'cosine'), (1.0, -1, 'EVM', 'cosine'),
           (1.0, 32, 'EVM', 'euclidean'), (
               0.1, 32, 'EVM', 'cosine'),
    (0.4, 128, 'EVM', 'cosine'), (0.7, 128, 'EVM', 'cosine'),
           (1.0, -1, 'EVM', 'cosine'),
    (1.0, 32, 'EVM', 'euclidean'),
(1.0, 32, 'EVM', 'euclidean'),
    (1.0, 32, 'EVM', 'euclidean'),
    (1.0, 32, 'EVM', 'euclidean')]


# Normally tqdm should capture it, i don't know y it doesn't
n_iterations = len(tail_sizes) * len(reduced_dims) * len(approaches) * len(distance_metrics)

good_configs = []

seed_everything(seed=13)

# for tail_size, reduced_dim, approach, distance_metric in tqdm.tqdm(
#     product(tail_sizes, reduced_dims, approaches, distance_metrics), total = n_iterations):
for tail_size, reduced_dim, approach, distance_metric in options:
    openset_options = {
        'tail_size': tail_size,
        'reduced_dim': reduced_dim,
        'approach': approach,
        'distance_metric': distance_metric
    }
    ret = simulate_noisy_rejection('Cars', 'LitInception', version=72, reference_epoch=28, batch_size=64,
                               openset_options=openset_options)
    if ret:
        # good_configs.append((tail_size, reduced_dim, approach, distance_metric))
        good_configs.append(openset_options)
        print('Found good config')

print('Good configs')
for gc in good_configs:
    print(gc)
