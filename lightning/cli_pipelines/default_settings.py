import os
import torch

PATH_DATASETS = os.environ.get("PATH_DATASETS", "/media/amidemo/Data/object_classifier_data/datasets")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
