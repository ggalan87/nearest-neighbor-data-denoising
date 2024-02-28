from pathlib import Path
import torch
from sklearn.decomposition import PCA

from features_storage import FeaturesStorage

from evt.vast_openset import *
from lightning.data.dataset_utils import random_split_perc
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from lightning_lite.utilities.seed import seed_everything


def load_test_features(dataset_name, model_name, version, reference_epoch):
    dataset_name = dataset_name
    model_class_name = model_name
    dm_name = dataset_name.lower()

    run_path = \
        Path(
            f'../lightning/cli_pipelines/lightning_logs/{dm_name}_{model_class_name}/version_{version}')

    cached_path_reference = run_path / 'features' / f'features_epoch-{reference_epoch}.pt'

    fs = FeaturesStorage(cached_path=cached_path_reference, target_key='target')

    # We entirely omit background images. In query images they do not exist but we do the same for completeness.
    (_, raw_test_feats), (_, raw_test_labels) = fs.raw_features()
    raw_testing_indices = fs.testing_feats['data_idx']
    raw_testing_is_noisy = fs.testing_feats['is_noisy']

    assert torch.count_nonzero(raw_testing_is_noisy) == 0

    return raw_test_feats, raw_test_labels, raw_testing_indices


seed_everything(13)

# dataset_name, model_name, version, reference_epoch
options = \
    {
        'dataset_name': 'Cars',
        'model_name': 'LitInception',
        'version': 72,
        'reference_epoch': 29,
    }

feats, labels, dataset_indices = load_test_features(**options)

openset_training_indices, openset_testing_indices = random_split_perc(len(feats), 0.5)

training_data = OpensetData(feats[openset_training_indices], labels[openset_training_indices],
                            dataset_indices[openset_training_indices])
testing_data = OpensetData(feats[openset_testing_indices], labels[openset_testing_indices],
                           dataset_indices[openset_testing_indices])


# Example configurations from https://github.com/Vastlab/vast/blob/main/vast/opensetAlgos/Example.ipynb
example_configurations = \
    [
        # ('OpenMax', "--distance_metric euclidean"),
        ('EVM', "--distance_metric cosine --tailsize 0.55 --distance_multiplier 0.7 --cover_threshold 0.7"),  # --cover_threshold
        ('EVM', "--distance_metric cosine --tailsize 0.55 --distance_multiplier 0.7 --cover_threshold 1.0"),
        # ('MultiModalOpenMax', "--distance_metric euclidean --tailsize 1.0 --Clustering_Algo dbscan"),
    ]

accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
accuracies = accuracy_calculator.get_accuracy(
    feats[openset_testing_indices], labels[openset_testing_indices], feats[openset_training_indices],
    labels[openset_training_indices]
)
print(f'Reference accuracy:{accuracies["precision_at_1"]}')


for (approach, algorithm_parameters) in example_configurations:
    saver_parameters = f"--OOD_Algo {approach}"
    model_params = OpensetModelParameters(approach, algorithm_parameters, saver_parameters)

    trainer = OpensetTrainer(training_data, model_params)

    trainer.train()
    trainer.eval_classification(testing_data)

    inclusion_probabilities = trainer.predict_proba(feats[openset_testing_indices])

    plt.figure()
    plt.matshow(inclusion_probabilities[:100], vmin=0, vmax=1)
    plt.colorbar()
    plt.title('test')
    plt.show()

