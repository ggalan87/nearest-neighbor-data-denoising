from pathlib import Path
import torch
from features_storage import FeaturesStorage
from evt.vast_openset import *

# embeddings_root = Path('../torch_metric_learning/embeddings')
embeddings_root = Path('../lightning/pipelines_datasets_scripts/mnist/features')


def load_plain_epoch_data(epoch):
    embeddings = torch.load(embeddings_root / f'train_embeddings_e{epoch}.pt')
    labels = torch.load(embeddings_root / f'train_labels_e{epoch}.pt')

    return OpensetData(features=embeddings, class_labels=labels)


def load_fs_epoch_data(epoch):
    cached_path = embeddings_root / f'dirtymnist_SimpleMNISTModel_epoch-{epoch}.pt'
    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    (raw_training_feats, raw_testing_feats), (raw_training_labels, raw_testing_labels) = fs.raw_features()

    print(f'Experiment with {len(torch.unique(raw_training_labels))} classes')

    training_data = OpensetData(raw_training_feats, raw_training_labels, None)
    testing_data = OpensetData(raw_testing_feats, raw_testing_labels, None)

    return training_data, testing_data


# train_data = load_plain_epoch_data(1)
# test_data = load_plain_epoch_data(2)

train_data, _ = load_fs_epoch_data(2)
test_data, _ = load_fs_epoch_data(3)


approach = 'OpenMax'
algorithm_parameters = "--distance_metric euclidean --tailsize 0.1"

# if approach == 'EVM':
#     algorithm_parameters += " --distance_multiplier 0.7"

saver_parameters = f"--OOD_Algo {approach}"
openset_model_params = OpensetModelParameters(approach, algorithm_parameters, saver_parameters)

# Reduce data
# train_embeddings = train_embeddings[random_indices]
# train_labels = train_labels[random_indices]
# random_indices = \
#     torch.randperm(len(train_labels))[:int(0.3 * len(train_labels))]

# Utilize a new instance of the openset trainer using the latest embeddings
openset_trainer = OpensetTrainer(train_data, openset_model_params, inference_threshold=0.5)

print('Openset training')
openset_trainer.train()

print('Openset evaluation')
openset_trainer.eval(test_data)
