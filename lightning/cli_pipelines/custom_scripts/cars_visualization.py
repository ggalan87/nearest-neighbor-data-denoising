from pathlib import Path
import csv
import numpy as np
from visualizations.embeddings_visualization import EmbeddingsVisualizer
from features_storage import FeaturesStorage
from lightning.data.datasets import Cars

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
    raw_training_is_noisy = fs.training_feats['is_noisy']

    training_feats = raw_training_feats

    return raw_training_feats, raw_training_labels, raw_training_indices, raw_training_is_noisy


dataset_name = 'Cars'
model_class_name = 'LitInception'
version = 72
reference_epoch = 28
training_variant = 'CARS_0.5noised'
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
        'data_idx': reference_indices
    }

fs = FeaturesStorage(dataset_name)
fs.add('train', dataset_attributes)

output_path = '/media/amidemo/Data/object_classifier_data/global_plots'
visualizer = EmbeddingsVisualizer(output_path, dataset_name=dataset_name)
cars_test_dataset = Cars('/media/amidemo/Data/object_classifier_data/datasets', train=True,
                         training_variant=training_variant)
visualizer.add_features(dataset_name, fs, datasets={'train': cars_test_dataset})
images_server = 'http://filolaos:8347'
visualizer.plot('bokeh', images_server=images_server)
