import copy
import pickle
from pathlib import Path

import numpy as np
import random
from collections import defaultdict

import torch
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from lightning.data.dataset_utils import batch_unpack_function
from lightning.ext import Singleton


class RandomIdentitySampler(Sampler, metaclass=Singleton):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), oid, camid, dsetid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size=16, num_instances=4, num_epochs=-1,
                 batch_unpack_fn=None, fix_samples=False, id_key='id', cached_oid_mapping: str =None):
        super().__init__(data_source)

        self._id_key = id_key
        # Fix samples for all calls of the sampler (training epochs). This is needed e.g. for also fixing the number
        # of steps as needed by some learning rate schedulers
        self._fix_samples = fix_samples
        self._num_epochs = num_epochs

        # 0. Check the parameters
        # if data_source is None or len(data_source) == 0:
        #     warn(f'{self.__class__.__name__} called without any data!')
        #     return
        self._parameters_check(batch_size, num_instances)

        # 1. Assign the parameters and compute useful variables from them
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_oids_per_batch = self.batch_size // self.num_instances
        if batch_unpack_fn is None:
            self.batch_unpack_fn = batch_unpack_function
        else:
            self.batch_unpack_fn = batch_unpack_fn

        # 2. Create a dictionary oid -> list of indices that correspond to the oid
        if cached_oid_mapping is not None and Path(cached_oid_mapping).exists():
            self.index_dic = pickle.load(open(cached_oid_mapping, 'rb'))
        else:
            self.index_dic = self._create_oid_mapping()
            if cached_oid_mapping is not None:
                pickle.dump(self.index_dic, open(cached_oid_mapping, 'wb'))

        # 3. Obtain the (unique) object ids
        self.oids = list(self.index_dic.keys())

        # 4. Check that there are enough oids for these settings
        assert len(self.oids) >= self.num_oids_per_batch, "Few ids of such settings"
        # Same as below -> assert num_instances * len(self.oids) > batch_size

        # 5. Estimate number of examples in an epoch. TODO: improve precision
        self.length = self._estimate_examples_per_epoch()

        if self._fix_samples:
            if self._num_epochs > 0:
                # +1 is for the case that dataloader is passed before actual training
                self._samples = [self._find_samples() for _ in range(self._num_epochs + 1)]
                self._n_iterations = 0

                # The only case in which we can compute the number of iterations is when number of epochs is known
                for epoch_idxs in self._samples:
                    self._n_iterations += int(len(epoch_idxs) / self.batch_size)
            else:
                self._samples = self._find_samples()
                self._n_iterations = -1
        else:
            self._samples = None
            self._n_iterations = -1

    @staticmethod
    def _parameters_check(batch_size, num_instances):
        # Just keep it simple, to avoid unnecessary checks and corner case bugs
        validity_predicates = \
            [
                batch_size >= num_instances,
                batch_size % num_instances == 0
            ]

        if not all(validity_predicates):
            raise ValueError(f'Validity predicate(s) failed', validity_predicates)

    def _create_oid_mapping(self):
        """
        Creates a mapping from object id to a list of samples indices corresponding to the id

        @return:
        """
        index_dic = defaultdict(list)

        print('Creating mapping of object ids to corresponding dataset indices')

        # Below I seek for existence of a data member which is the actual underlying dataset without the image
        # I do so to avoid the extra overhead of directly calling the __get_item__() method which normally involves
        # loading of the corresponding image, which is a harshly slow approach
        if hasattr(self.data_source, 'data'):
            dataset = self.data_source.data
        # The case below is error prone because it matched with Subset class, whose "data" attribute is the entire
        # dataset rather than the subset, and in turn invalid ids are computed. More logic is needed to support this
        # elif hasattr(self.data_source, 'dataset') and hasattr(self.data_source.dataset, 'data'):
        #     dataset = self.data_source.dataset.data
        else:
            dataset = self.data_source

        # dataset = self.data_source

        for index, ann in tqdm(enumerate(dataset), total=len(dataset)):
            # Since the returned item is a tuple, we simply obtain the first tuple element
            oid = self.batch_unpack_fn(ann, keys=(self._id_key, ))[0]
            if isinstance(oid, torch.Tensor):
                oid = oid.item()
            index_dic[oid].append(index)
        return index_dic

    def _estimate_examples_per_epoch(self):
        length = 0
        for oid in self.oids:
            idxs = self.index_dic[oid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            length += num - num % self.num_instances

        return length

    def _find_samples(self):
        # TODO: shuffle within batch ids
        batch_idxs_dict = defaultdict(list)

        for oid in self.oids:
            idxs = copy.deepcopy(self.index_dic[oid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[oid].append(batch_idxs)
                    batch_idxs = []

        avai_oids = copy.deepcopy(self.oids)
        final_idxs = []

        while len(avai_oids) >= self.num_oids_per_batch:
            selected_oids = random.sample(avai_oids, self.num_oids_per_batch)
            for oid in selected_oids:
                batch_idxs = batch_idxs_dict[oid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[oid]) == 0:
                    avai_oids.remove(oid)

        return final_idxs

    def __iter__(self):
        # Not stored
        if self._samples is None:
            final_idxs = self._find_samples()
        # List per epoch
        elif isinstance(self._samples, list) and isinstance(self._samples[0], list):
            final_idxs = self._samples.pop(0)
        else:
            final_idxs = self._samples

        return iter(final_idxs)

    def __len__(self):
        return self.length

    @property
    def n_iterations(self):
        return self._n_iterations
