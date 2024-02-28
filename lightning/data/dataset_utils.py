from pathlib import Path
from typing import Tuple, Dict, Optional, List
import random
import torchvision.datasets
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def generate_random_keep(n_samples: int, keep_probability: float):
    """
    Generates an array of length n_samples which has keep_probability * n_samples ones and the rest are zeros at random
    positions.

    @param n_samples:
    @param keep_probability:
    @return:
    """
    # Adapted from: https://stackoverflow.com/a/19597672
    n_ones = int(keep_probability * n_samples)
    arr = np.zeros(n_samples)
    arr[:n_ones] = 1
    np.random.shuffle(arr)
    return arr


def batch_unpack_function(batch_dict, keys=('image', 'target', 'data_idx')):
    """
    This is a function which unpacks the dictionary entry that is returned from the dataset from its __get_item__()
    method. In ordinary implementations __get_item__() just returns a tuple with images and labels, however for more
    complex stuff we need to return more things. An example is the data_idx which is the index to the data list, which
    for example is useful for pointing to the original image for visualization purposes afterwards. image field contains
    the transformed one which is not useful for such scope but only for feeding the neural network. The given keys must
    exist in the batch dictionary. This is up to the user of the function.

    I cannot standardize things because conceptually labels are called ids in re-id datasets, or targets or anything
    else in other contexts or according to the preference of the programmer / framework / library.

    I prefer singular form (e.g. image and not images) even if the underlying thing is eventually collated data just to
    be consistent with the single element dictionaries which are contained in the raw data list.

    Example usages:
    (a) Unpack image, target, and data_idx into x, y, z variables (default keys)
    x, y, z = batch_unpack_function(batch)

    (b) Unpack image and target into x, y variables
    x, y = batch_unpack_function(batch, keys=('image', 'target'))

    (c) Unpack a field called id a corresponding variable
    batch_ids = batch_unpack_function(batch, keys=('id'))


    @param batch_dict: The dictionary which contins the batch data, i.e. collated (tensors of) images, labels etc
    @param keys: The keys to extract in the tuple
    @return: A tuple containing the unpacked dictionary
    """
    return tuple(batch_dict[k] for k in keys)


def export_binary_predicted_images(dataset: torchvision.datasets.VisionDataset,
                                   predictions: torch.Tensor,
                                   dataset_indices: torch.Tensor, keep_ratio=0.005,
                                   output_path: str = './output_data',
                                   binary_names: Tuple = ('included', 'not included')):
    """
    Given predictions and indices of these predictions which correspond to the original dataset, it saves images into
    separate folders for visual inspection of the predictions. Indices are given because predictions might not be in
    ascending indices order which is the case for randomization in training data due to sampling which might not have
    been muted.

    @param dataset: An image dataset
    @param predictions: Binary predictions
    @param dataset_indices: Corresponding dataset indices of these predictions
    @param keep_ratio: Ratio to keep subset of images in the dataset
    @param output_path: Path to save the results
    @param binary_names: Names for the individual folders in the form of (positive name, negative name)
    @return: None
    """
    assert 0 < len(torch.unique(predictions) < 3)
    assert dataset_indices.max() < len(dataset.data)

    keep_indices = generate_random_keep(n_samples=len(dataset_indices), keep_probability=keep_ratio)
    predictions = predictions[keep_indices == 1]
    dataset_indices = dataset_indices[keep_indices == 1]

    positive_class_path = Path(output_path) / binary_names[0]
    negative_class_path = Path(output_path) / binary_names[1]
    train_tag = f'train={str(dataset.train)}'
    image_tag = f'{dataset.__class__.__name__}_{train_tag}'

    print('Saving images...')
    for i in tqdm(range(len(dataset_indices))):
        idx = dataset_indices[i]
        im = Image.fromarray(dataset.data[idx].numpy())

        if predictions[i] == 1:
            im.save(positive_class_path / f'{image_tag}_{idx}.jpg')
        elif predictions[i] == 0:
            im.save(negative_class_path / f'{image_tag}_{idx}.jpg')
        else:
            assert False


def disturb_targets_symmetric(targets: torch.Tensor, perc: float):
    # Find the unique class labels
    unique_labels = torch.unique(targets).tolist()

    # Get a clone of the original targets
    noisy_targets = targets.clone()

    disturbed_indices_list = []

    # Iterate the labels in a list such that we get integers and not tensors
    for l in unique_labels:
        other_labels = \
            torch.tensor(list(set(unique_labels) - {l}))

        # Find the indices of those labels in the entire labels tensor
        l_indices = torch.where(targets == l)[0]
        # Create random permutation of a range of integers (0...length-1)
        perm = torch.randperm(len(l_indices))
        # Find the actual number of indices to disturb, subset of all corresponding to this label
        n_indices_to_disturb = int(perc * len(l_indices))
        # Select the randomly generated indices
        indices_to_disturb = l_indices[perm[:n_indices_to_disturb]]
        # Select random labels from the pool of other labels
        disturbed_labels = random.choices(other_labels, k=n_indices_to_disturb)
        # Assign these labels to labels tensor
        noisy_targets[indices_to_disturb] = torch.tensor(disturbed_labels, dtype=targets.dtype)
        disturbed_indices_list.append(indices_to_disturb)

    indices = torch.hstack(disturbed_indices_list) if len(disturbed_indices_list) > 0 else None

    noisy_indices = torch.zeros((len(noisy_targets), ), dtype=torch.bool)
    noisy_indices[indices] = True
    return noisy_targets,  noisy_indices


def disturb_targets(targets: torch.Tensor, labels_noise_perc: Dict) \
        -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Iterates all unique class labels and if the label is encountered in labels_noise_perc dictionary, part of
    the indices (or all) are randomly assigned to another label.

    @return: A tensor containing the noisy indices
    """

    # Find the unique class labels
    unique_labels = torch.unique(targets)

    # Get a clone of the original targets
    noisy_targets = targets.clone()

    # Find the rest labels (except those included in the option)
    other_labels = \
        torch.tensor(list(set(unique_labels.tolist()) - set(labels_noise_perc.keys())))

    disturbed_indices_list = []

    # Iterate the labels in a list such that we get integers and not tensors
    for l in unique_labels.tolist():
        # If the label is encountered in the dict we do the disturbance.
        # I do it like this to also prevent wrong labels, else iteration of the dictionary keys is better option
        if l in labels_noise_perc:
            # Find the indices of those labels in the entire labels tensor
            l_indices = torch.where(targets == l)[0]
            # Create random permutation of a range of integers (0...length-1)
            perm = torch.randperm(len(l_indices))
            # Find the actual number of indices to disturb, subset of all corresponding to this label
            n_indices_to_disturb = int(labels_noise_perc[l] * len(l_indices))
            # Select the randomly generated indices
            indices_to_disturb = l_indices[perm[:n_indices_to_disturb]]
            # Select random labels from the pool of other labels
            disturbed_labels = random.choices(other_labels, k=n_indices_to_disturb)
            # Assign these labels to labels tensor
            noisy_targets[indices_to_disturb] = torch.tensor(disturbed_labels, dtype=targets.dtype)
            disturbed_indices_list.append(indices_to_disturb)

            if labels_noise_perc[l] == 1.0 and torch.count_nonzero(noisy_targets == l) != 0:
                raise AssertionError(f'At least on label {l} left in labels. Disturbance logic error!')

    for k, v in labels_noise_perc.items():
        if v == 1.0 and len(noisy_targets[noisy_targets == k]) != 0:
            raise AssertionError(f'Disturbance logic error. '
                                 f'One or more of noisy labels with probability 1.0 in {list(labels_noise_perc.keys())} '
                                 f'exist in targets.')

    indices = torch.hstack(disturbed_indices_list) if len(disturbed_indices_list) > 0 else None
    if indices is None:
        return None, None

    noisy_indices = torch.zeros((len(noisy_targets), ), dtype=torch.bool)
    noisy_indices[indices] = True
    return noisy_targets,  noisy_indices


def disturb_records_targets(data: List[Dict], labels_noise_perc: Dict, key_to_disturb='target'):
    """
    Some datasets do not contain a separate targets tensor but instead contain all the data in the form of records, i.e.
    list of dicts. This essentially transforms this kind of data into pandas dataframe for easier access to underlying
    targets data and then calls the original function to manipulate the targets.

    @param data:
    @param labels_noise_perc:
    @param key_to_disturb:
    @return:
    """
    data_df = pd.DataFrame(data)
    targets = torch.tensor(data_df[key_to_disturb].values)

    # Special case for symmetric noise
    if 'symmetric' in labels_noise_perc:
        noisy_targets, noisy_indices = disturb_targets_symmetric(targets, labels_noise_perc['symmetric'])
    else:
        noisy_targets, noisy_indices = disturb_targets(targets, labels_noise_perc)
    if noisy_indices is None:
        return

    data_df.loc[:, 'target'] = noisy_targets.numpy()
    data_df['is_noisy'] = noisy_indices.numpy()

    return data_df.to_dict('records')


def random_split_perc(data_length: int, split_percentage: float = 0.5):
    # Create random permutation of a range of integers (0...length-1)
    perm = torch.randperm(data_length)
    # Find the split point
    split_point = int(split_percentage * data_length)

    return perm[:split_point], perm[split_point:]


def pack_decoupled_data(**kwargs):
    """
    Given separate dataset elements, e.g. data, targets, etc merges them into an alternative view of as single list of
    dictionaries

    @param kwargs:
    @return: List of dictionaries
    """
    data_items = []

    for elements in zip(*list(kwargs.values())):
        data_item = {}
        for i, k in enumerate(kwargs.keys()):
            elem = elements[i]
            if isinstance(elem, torch.Tensor) and elem.ndim == 0:
                data_item[k] = elem.item()
            else:
                data_item[k] = elem

        data_items.append(data_item)
    return data_items

