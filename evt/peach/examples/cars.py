from PEACH import PEACH as PEACH_Algo

import numpy as np
import h5py
import torch
import csv
from sklearn import metrics
import pickle
from pathlib import Path

from features_storage import FeaturesStorage

version = 1
epoch = -1
cached_path = \
    Path(f'/media/amidemo/Data/object_classifier_data/logs/lightning_logs/cars_LitInception/version_{version}/features/'
         f'features_epoch-{epoch}.pt')
fs = FeaturesStorage(cached_path=cached_path, target_key='target')

# We entirely omit background images. In query images they do not exist but we do the same for completeness.
(raw_training_feats, raw_testing_feats), (raw_training_labels, raw_testing_labels) = fs.raw_features()

features = raw_testing_feats
result = PEACH_Algo(features, 0, no_singleton=False, metric="cosine", batch_size=4096, evt=True)  # 0 means GPU0
y_pred = np.array(result)
