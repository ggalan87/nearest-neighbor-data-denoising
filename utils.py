import psutil
from typing import Callable, Dict
import inspect
from timeit import default_timer as timer
import numpy as np
import torch
from torch.utils.data import DataLoader
import importlib


def compute_dataset_mean_std(dataloader: DataLoader):
    # https://stackoverflow.com/a/59182940/1281111
    mean = 0.
    mean_square = 0.
    samples = 0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        mean_square += (images**2).mean(2).sum(0)
        samples += images.size(2) * images.size(0)

    mean /= len(dataloader.dataset)
    mean_square /= len(dataloader.dataset)

    # extra scale factor for unbias std estimate (it's effectively 1.0)
    scale = samples / (samples - 1)
    std = torch.sqrt((mean_square - mean**2) * scale)

    return mean, std


def report_mem_usage():
    mem = psutil.virtual_memory()
    swap_mem = psutil.swap_memory()

    print(f'Memory usage: {100 * mem.used / mem.total:.2f}, '
          f'Swap Memory usage: {100 * swap_mem.used / swap_mem.total:.2f}')


def count_populations(all_labels: torch.Tensor):
    populations = torch.bincount(all_labels)
    ids = torch.arange(len(populations))
    ids = ids[populations != 0]
    populations = populations[populations != 0]

    return populations, ids


def measure_time(message: str, fun: Callable, **kwargs):
    start = timer()

    ret = fun(**kwargs)

    end = timer()
    print(f'{message}. Elapsed {end - start} sec.')
    return ret


def merge(a: Dict, b: Dict, path=None):
    """ Merges two dictionaries recursively """
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def class_from_string(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    klass = getattr(importlib.import_module(module_name), class_name)
    if inspect.isclass(klass):
        return klass
    else:
        raise ValueError(f'{class_path} does not represent the path of a class.')
