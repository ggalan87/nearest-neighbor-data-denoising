from warnings import warn
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, Dict, List
import numpy as np
import torch


class FeaturesStorage:
    """
    A simple class to handle saving of features in a structured (dictionary based) file.
    TODO: Consider subclassing collections.UserDict

    Keys that MUST be used in leaf dictionary:
    feats: the NxD features
    target: Nx1 groundtruth label

    Keys that MAY be used in leaf dictionary:
    dataset_idx: Nx1 giving the index to the dataset
    predicted_target: Nx1 predicted label


    Sample format:
    {
      'info':
          {
            'dataset_name': dataset_name,
            'description': ''
          },
      'feats': {
            'trainval':
            {
                'feats': trainval_feats,
                'target': trainval_labels,
                <other-key>: <other-value>
                ...
            },
            'test':
            {
                'feats': test_feats,
                'target': test_labels
                <other-key>: <other-value>
                ...
            }
    }
    """
    def __init__(self, dataset_name=None, cached_path=None, target_key: str = 'target'):
        # We need XNOR case, for either new dataset or loading existing one
        if not((dataset_name is None) ^ (cached_path is None)):
            raise AssertionError(
                'You either need to specify a dataset name for a new storage instance or a path for loading existing '
                'features')

        if cached_path is None:
            self._features_dict = \
                {
                    'info':
                        {
                          'dataset_name': dataset_name,
                          'description': ''
                        },
                    'feats': {}
                }
        else:
            self._features_dict = torch.load(cached_path)

        self._target_key = target_key
        self._valid_parts = ['train', 'trainval', 'val', 'test']

        self._kept_indices = None

    def add(self, part: str, dataset_attributes: Dict[str, Union[torch.Tensor, np.ndarray]]):
        """
        Verifies and stores a dictionary of dataset attributes (features, corresponding labels, etc) from all part
        samples. The elements of the dictionary correspond to a concatenation of a pass on dataloader, therefore
        normally whatever dataloader is told to return on every batch, should be passed here. Dataloader is the defacto
        way to iterate over datasets, however manual dataset loadnig/iteration would also work.

        @param part: which dataset part (train, trainval, val, test)
        @param dataset_attributes: dictionary str->tensor which contains the features, targets (labels), data indices,
        and other possible image info that will be implemented, e.g. camera id, object id etc
        @return:
        """
        def ensure_tensor(var):
            if isinstance(var, np.ndarray):
                return torch.from_numpy(var)
            else:
                return var

        if part not in self._valid_parts:
            raise ValueError(f'Invalid dataset part {part}. Should be one of {self._valid_parts}')
        if part in self._features_dict:
            warn('Updating features of existing part.')

        # Ensure that all values in the dict are tensors, or convert them otherwise and fetch to CPU if not there
        # Also keep the lengths to ensure that all are of same size
        sizes = []
        for k, v in dataset_attributes.items():
            dataset_attributes[k] = ensure_tensor(v).cpu().detach()
            sizes.append(len(dataset_attributes[k]))

        if not all([sizes[0] == s for s in sizes]):
            raise ValueError('All inputs must be of same length')

        self._features_dict['feats'][part] = dataset_attributes

    def save(self, filepath: str):
        output = Path(filepath)
        output.parents[0].mkdir(parents=True, exist_ok=True)
        torch.save(self._features_dict, output)

    @property
    def training_feats(self) -> Optional[Dict[str, torch.Tensor]]:
        # We look for train or trainval in that order and return the first encounter
        possible_tags = ['train', 'trainval']
        for pt in possible_tags:
            if pt in self._features_dict['feats']:
                return self._features_dict['feats'][pt]
        return None

    @property
    def testing_feats(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._features_dict['feats'].get('test', None)

    @property
    def possible_parts(self):
        return self._valid_parts

    def __getitem__(self, item):
        return self._features_dict[item]

    def raw_features(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """ Packs and returns features and labels in two seperate lists for easier access """
        def try_get(dict_obj, dict_item):
            try:
                return dict_obj[dict_item]
            except TypeError:
                return None

        training_feats = self.training_feats
        testing_feats = self.testing_feats

        return [try_get(training_feats, 'feats'), try_get(testing_feats, 'feats')], \
               [try_get(training_feats, self._target_key), try_get(testing_feats, self._target_key)]

    def get_parts_tags(self) -> List[str]:
        """ Returns a list with the existing dataset parets in the storage """
        return list(self._features_dict['feats'].keys())

    @property
    def target_key(self):
        return self._target_key

    @property
    def kept_indices(self):
        return self._kept_indices

    def filter_by_ids(self, keep_range: Tuple[int, int]):
        # First, rename the dataset, such that it cannot be messed up with the original one
        orig_name = self._features_dict['info']['dataset_name']
        self._features_dict['info']['dataset_name'] = f'{orig_name}_filtered_subset'

        self._kept_indices = {}

        # Second do the filtering for each part / same filtering
        for part, feats_info in self._features_dict['feats'].items():
            labels = feats_info[self.target_key]
            feats = feats_info['feats']
            keep_indices = torch.bitwise_and(labels >= keep_range[0], labels <= keep_range[1])

            self._features_dict['feats'][part]['feats'] = feats[keep_indices]
            self._features_dict['feats'][part][self.target_key] = labels[keep_indices]

            # Store kept indices for possible utilization externally
            self._kept_indices[part] = keep_indices
