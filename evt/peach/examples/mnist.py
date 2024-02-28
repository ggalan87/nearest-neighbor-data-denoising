from PEACH import PEACH as PEACH_Algo

import numpy as np
import h5py
import torch
import csv
from sklearn import metrics
import pickle
#data = np.random.rand(100,5)
from pathlib import Path

########Get data###############
from torchvision import datasets, transforms
h = h5py.File('../PEACH/mnist_LeNet.hdf5')

features = h["features"]
features = np.array(features, dtype=float)

test_loader = torch.utils.data.DataLoader(
          datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
          ])), shuffle=False)
subjects = [int(target.cpu().numpy()) for data, target in test_loader]
count = 0
##############################
######use FACTO###############

with_evt = True
if with_evt:
    predictions_filepath = 'mnist_pred_evt.npy'
else:
    predictions_filepath = 'mnist_pred.npy'

if not Path(predictions_filepath).exists():
    result = PEACH_Algo(features, 0, no_singleton=False, metric="cosine", batch_size=4096, evt=with_evt) # 0 means GPU0
    y_pred = np.array(result)
    np.save(predictions_filepath, y_pred)
else:
    y_pred = np.load(predictions_filepath)

y_true = np.array(subjects)

for l in np.unique(y_true):
    l_indices = np.where(y_true == l)[0]
    pred_for_l = y_pred[l_indices]
    print(np.count_nonzero(pred_for_l == -1) / len(pred_for_l))
    pred_for_l = pred_for_l[pred_for_l > -1]
    # pred_for_l += 1
    counts = np.bincount(pred_for_l)
    print(f'label {l} was spread into clusters {np.unique(pred_for_l)},'
          f' but most of them ({np.max(counts)}) were in {np.argmax(counts)}')


NMI = metrics.normalized_mutual_info_score(y_true, y_pred, average_method='max')
print("NMI:", NMI)