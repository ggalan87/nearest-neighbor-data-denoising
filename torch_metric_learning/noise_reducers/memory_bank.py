from abc import abstractmethod
from warnings import warn
import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import CrossBatchMemory
from pytorch_metric_learning.utils import common_functions as c_f

class MemoryBase:
    def __init__(self):
        self._memory = None

    @abstractmethod
    def add_to_memory(self, embeddings, labels, dataset_indices=None) -> None:
        pass

    @abstractmethod
    def reset_memory(self):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @property
    def memory(self):
        return self._memory


class DynamicMemory(MemoryBase):
    def __init__(self):
        super().__init__()

        self._memory = \
            {
                'features': [],
                'labels': [],
                'dataset_indices': []
            }

    def add_to_memory(self, embeddings, labels, dataset_indices=None):
        self._memory['features'].append(embeddings)
        self._memory['labels'].append(labels)

        if dataset_indices is not None:
            self._memory['dataset_indices'].append(dataset_indices)

    def reset_memory(self):
        for l in self._memory.values():
            l.clear()

    def get_data(self):
        all_features = torch.vstack(self._memory['features'])
        all_class_labels = torch.hstack(self._memory['labels'])
        all_indices = torch.hstack(self._memory['dataset_indices']) \
            if len(self._memory['dataset_indices']) > 0 else None

        return all_features, all_class_labels, all_indices


class PreallocatedMemory(MemoryBase):
    def __init__(self, preallocated_object: CrossBatchMemory = None):
        super().__init__()

        self._memory = preallocated_object

    def add_to_memory(self, embeddings, labels, dataset_indices=None):
        if dataset_indices is not None:
            warn('CrossBatchMemory object does not support saving dataset indices, but only embeddings and labels')

        # Copied from CrossBatchMemory forward()
        device = embeddings.device
        labels = c_f.to_device(labels, device=device)
        self._memory.embedding_memory = c_f.to_device(
            self._memory.embedding_memory, device=device, dtype=embeddings.dtype
        )
        self._memory.label_memory = c_f.to_device(
            self._memory.label_memory, device=device, dtype=labels.dtype
        )

        self._memory.add_to_memory(embeddings, labels, batch_size=len(embeddings))

    def reset_memory(self):
        warn('Normally the memory is never reset but works as FIFO queue Is this intentional?')
        self._memory.reset_queue()

    def get_data(self):
        if not self._memory.has_been_filled:
            embeddings = self._memory.embedding_memory[: self._memory.queue_idx]
            labels = self._memory.label_memory[: self._memory.queue_idx]
        else:
            embeddings = self._memory.embedding_memory
            labels = self._memory.label_memory

        return embeddings, labels, None


class MemoryBank:
    """
    A memory bank class which supports either dynamic preallocated memory. Dynamic memory is essentially a set of lists
    which are progressively filled with tensor data, e.g. embeddings and labels, while preallocated memory is a buffer
    which is pre-allocated with a specific size and works as a FIFO queue afterwards. Preallocated memory implementation
    is a wrapper class around CrossBatchMemory class. This is because in some cases this type of loss wrapper utilizes
    such memory bank, and therefore I opted to have a shared memory. Essentially if tensors are not copied this should
    be the case with dynamic memory, too.

    Currently I don't support creation of CrossBatchMemory object if this does not exist
    """
    def __init__(self, with_dynamic_memory: bool = True, preallocated_object: CrossBatchMemory = None):

        self._with_dynamic_memory = with_dynamic_memory

        if self._with_dynamic_memory and preallocated_object is not None:
            raise AssertionError('A preallocated object can be passed only when preallocated type is specified.')
        elif not self._with_dynamic_memory and preallocated_object is None:
            raise NotImplementedError

        self._memory_object = DynamicMemory() if self._with_dynamic_memory else PreallocatedMemory(preallocated_object)

    def get_data(self, samples_fraction=1.0, do_reset_memory=True):
        """
        this method causes also a release of memory

        @return:
        """

        # Retrieve all data from memory
        all_features, all_class_labels, all_indices = self._memory_object.get_data()

        # Possibly get a fraction of the data
        if samples_fraction < 1.0:
            random_indices = \
                torch.randperm(len(all_class_labels))[:int(samples_fraction * len(all_class_labels))]
        else:
            random_indices = torch.arange(len(all_class_labels))

        all_features = all_features[random_indices]
        all_class_labels = all_class_labels[random_indices]

        # Normalize the features
        all_features = F.normalize(all_features)

        # Release the memory bank to avoid memory issues afterwards
        if do_reset_memory:
            self.reset_memory()

        return all_features, all_class_labels, all_indices, random_indices

    def add_to_memory(self, embeddings, labels, dataset_indices=None):
        self._memory_object.add_to_memory(embeddings, labels, dataset_indices)

    def reset_memory(self):
        # Pre-allocated memory works as FIFO and must not be released. If needed one has to explicitly call the
        # underlying object method
        if isinstance(self._memory_object, DynamicMemory):
            self._memory_object.reset_memory()

    @property
    def with_dynamic_memory(self):
        return self._with_dynamic_memory

    @property
    def mem_object(self):
        return self._memory_object
