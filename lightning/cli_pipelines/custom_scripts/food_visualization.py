from pathlib import Path
import csv
import numpy as np
from visualizations.embeddings_visualization import EmbeddingsVisualizer
from features_storage import FeaturesStorage
from lightning.data.datasets import Food101N

# dataset_name = 'cars196'
# features_path = '/home/amidemo/devel/workspace/Ranking-based-Instance-Selection/output/2023-01-07_feat.npz'
# npz_obj = np.load(features_path)
#
# dataset_attributes = \
#     {
#         'feats': npz_obj['feat'],
#         'target': npz_obj['upc'].astype(np.int32),
#         'data_idx': np.arange(len(npz_obj['feat']))
#     }


def load_data(cached_path, samples_fraction=1.0):
    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    # We entirely omit background images. In query images they do not exist but we do the same for completeness.
    (raw_training_feats, _), (raw_training_labels, _) = fs.raw_features()
    raw_training_indices = fs.training_feats['data_idx']

    try:
        raw_training_is_noisy = fs.training_feats['is_noisy']
    except KeyError:
        raw_training_is_noisy = None

    training_feats = raw_training_feats

    return raw_training_feats, raw_training_labels, raw_training_indices, raw_training_is_noisy


dataset_name = 'Food101N'
model_class_name = 'LitUnicom'
version = 0
reference_epoch = -1
training_variant = 'verified'
dm_name = dataset_name.lower()

run_path = \
    Path(
        f'/media/amidemo/Data/object_classifier_data/logs/lightning_logs/{dm_name}_{model_class_name}/version_{version}')

cached_path_reference = run_path / 'features' / f'features_epoch-{reference_epoch}.pt'

reference_feats, reference_labels, reference_indices, reference_is_noisy = \
    load_data(cached_path_reference)

dataset_attributes = \
    {
        'feats': reference_feats,
        'target': reference_labels,
        'data_idx': reference_indices,
        #'is_noisy': reference_is_noisy
    }

fs = FeaturesStorage(dataset_name)
fs.add('train', dataset_attributes)

output_path = '/media/amidemo/Data/object_classifier_data/global_plots'
visualizer = EmbeddingsVisualizer(output_path, dataset_name=dataset_name)
food101n_train_dataset = Food101N('/media/amidemo/Data/object_classifier_data/datasets', train=True,
                         training_variant=training_variant)
visualizer.add_features(dataset_name, fs, datasets={'train': food101n_train_dataset})
images_server = 'http://filolaos:8347'
visualizer.plot('bokeh', images_server=images_server)
