from pathlib import Path
import csv
import numpy as np
from features_storage import FeaturesStorage
from lightning.data.datasets import Cars98N
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

for p in Path('/home/workspace/Ranking-based-Instance-Selection/output/CARS_feats').iterdir():
    if '.npz' not in p.name:
        continue

    dataset_name = 'cars196'
    features_path = str(p)
    print(features_path)

    npz_obj = np.load(features_path)

    dataset_attributes = \
        {
            'feats': npz_obj['feat'],
            'target': npz_obj['upc'].astype(np.int32)
        }

    fs = FeaturesStorage(dataset_name)
    fs.add('test', dataset_attributes)

    (_, raw_testing_feats), (_, raw_testing_labels) = fs.raw_features()

    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    accuracies = accuracy_calculator.get_accuracy(raw_testing_feats, raw_testing_labels)
    print(accuracies["precision_at_1"])

