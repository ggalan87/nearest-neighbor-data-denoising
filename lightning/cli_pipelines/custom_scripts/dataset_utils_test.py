import torch

from lightning.data import dataset_utils as du

labels = [0] * 10 + [1] * 10 + [2] * 10
labels = torch.Tensor(labels)
labels_noise_perc = {0: 0.5, 1: 0.5, 2: 0.5}
print(labels)

print(du.disturb_targets_symmetric(labels, 0.5))
