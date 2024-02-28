import copy
import numpy as np
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
from functools import reduce
import operator


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), oid, camid, dsetid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size=16, num_instances=4):
        super().__init__(data_source)

        # Just keep it simple, to avoid unnecessary checks and corner case bugs
        validity_predicates = \
            [
                batch_size >= num_instances,
                batch_size % num_instances == 0
            ]

        if not all(validity_predicates):
            raise ValueError(f'Validity predicate(s) failed', validity_predicates)

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_oids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, ann in enumerate(data_source.data_infos):
            oid = ann['gt_label'].item()
            self.index_dic[oid].append(index)

        self.oids = list(self.index_dic.keys())

        # Same as below -> assert num_instances * len(self.oids) > batch_size
        assert len(self.oids) >= self.num_oids_per_batch, "Few ids of such settings"

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0

        for oid in self.oids:
            idxs = self.index_dic[oid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        # print(self.length)

    def __iter__(self):
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

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomClassIdentitySampler(Sampler):
    """A sampler to support losses which require both identities and classes. It randomly samples N identities each with K instances. For each identity

    Args:
        data_source (list): contains tuples of (img_path(s), oid, camid, dsetid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size=16, objects_per_class=2, num_instances=4):
        # Just keep it simple, to avoid unnecessary checks and corner case bugs
        validity_predicates = \
            [
                batch_size > num_instances,
                batch_size % num_instances == 0
            ]
        if not all(validity_predicates):
            raise ValueError(f'Validity predicate(s) failed', validity_predicates)

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances # How many instamces of the same identity/subclass will be in the batch
        self.objects_per_class = objects_per_class  # How many objects of the same class are inside the batch, e.g. for #2, A1,A2,B1,B2
        self.num_oids_per_batch = self.batch_size // self.num_instances  # Total number of identities/subclasses in the batch
        self.num_classes_per_batch = min(self.num_oids_per_batch // self.objects_per_class, len(self.data_source.super_classes))  # How many different classes will be in the batch

        # object id -> list of indices in the original dataset
        self.index_dic = defaultdict(list)
        for index, ann in enumerate(data_source.data_infos):
            oid = ann['gt_label'].item()
            self.index_dic[oid].append(index)

        # list of all ids
        self.oids = list(self.index_dic.keys())

        assert num_instances * len(self.oids) > batch_size, "Few ids of such settings"

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0

        for oid in self.oids:
            idxs = self.index_dic[oid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        #print(self.length)
        #print(self.data_source.super_classes)

    def __iter__(self):
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

        avai_classes = copy.deepcopy(self.data_source.super_classes)
        avai_oids = copy.deepcopy(self.oids)
        final_idxs = []

        # 2nd condition: availability from at least 'num_classes_per_batch', if less then stop
        while len(avai_oids) >= self.num_oids_per_batch and self.num_classes_per_batch <= len(avai_classes):
            # Subset of the super_classes list (list of lists), where each inner list corresponds to a superclass
            # Here we randomly select which superclasses will be represented in the batch
            selected_class = copy.deepcopy(random.sample(avai_classes, self.num_classes_per_batch))

            # Here we select which subclasses will be represented in the batch
            selected_oids = []
            for il in selected_class:
                # Randomize the subclasses
                random.shuffle(il)
                # Select first few
                selected_oids.extend(il[:self.objects_per_class])

            for oid in selected_oids:
                batch_idxs = batch_idxs_dict[oid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[oid]) == 0:
                    avai_oids.remove(oid)
                    for e in avai_classes:
                        if oid in e:
                            avai_classes.remove(e)
                            break

        return iter(final_idxs)

    def __len__(self):
        return self.length