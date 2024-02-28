from abc import abstractmethod
import torch

from .memory_bank import MemoryBank
from .sample_rejection import RejectionStrategy
from .sample_rejection.rejection_base import Inspector


class DefaultNoiseReducer(torch.nn.Module):
    def __init__(self, strategy: RejectionStrategy, memory_bank, warm_up_epochs: int = 0, keep_only_good_samples=False,
                 use_pretrained=False, with_inspector=False):
        """

        @param strategy: a rejection strategy, which realizes if a sample is  noisy or not
        @param memory_size: 0 means dynamic memory, above zero means static preallocated memory
        @param warm_up_epochs: number of starting epochs to mute the noise reducer
        @param keep_only_good_samples: whether to store only clean samples to memory
        @param use_pretrained: use pretrained network to initialize the noise reduction before training starts
        """
        super().__init__()

        if warm_up_epochs > 0 and use_pretrained:
            raise ValueError('You required to use pretrained network weights with warm up. This will have no effect!')

        self._strategy = strategy
        self._keep_only_good_samples = keep_only_good_samples
        self._use_pretrained = use_pretrained
        self.current_epoch = 0
        self.warm_up_epochs = warm_up_epochs

        self._memory_bank = memory_bank
        self._inspector = Inspector() if with_inspector else None

    def forward(self, embeddings, labels, noisy_samples=None):
        """

        @param embeddings: the embeddings of th batch
        @param labels: the labels of the batch
        @param noisy_samples: ground truth noisy samples, used for debugging/evaluation
        @return:
        """
        self._strategy.current_batch_noisy_samples = noisy_samples
        return self._noise_reduction_impl(embeddings, labels)

    def bootstrap_initial(self, embeddings, labels):
        """ Can be called before starting training to initialize with pretrained weights """
        print('Using vanilla features for training...')
        self._memory_bank.add_to_memory(embeddings, labels)
        self._strategy.train(self._memory_bank)

    def bootstrap_epoch(self, epoch=-1):
        """
        This function should be called inside training_epoch_end() hook. In this case the epoch has already
        incremented by 1.
        TODO: Consider altering logic for epoch_start / there is no such hook in lightning (1.8.x)

        @param epoch:
        @return:
        """
        if self.current_epoch >= self.warm_up_epochs:
            if self._inspector is not None:
                self._inspector.report_and_reset(self._strategy.labels_to_indices)
            self._strategy.train(self._memory_bank)
        else:
            # TODO: since refactoring we know inside this class when to fill the memory, therefore we can omit this
            #  call
            self._memory_bank.reset_memory()

        if epoch == -1:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch

    @property
    def use_pretrained(self):
        return self._use_pretrained

    @property
    def inspector(self):
        return self._inspector

    @property
    def strategy(self):
        return self._strategy

    def memory_is_dynamic(self):
        return self._memory_bank.with_dynamic_memory

    def has_trained(self):
        return self._strategy.has_trained()

    def _noise_reduction_impl(self, embeddings, labels):
        # TODO: Consider move this logic inside rejection strategy
        if self._strategy.has_trained():
            batch_noisy_predictions = self._strategy.predict_noise(embeddings, labels)
        else:
            batch_noisy_predictions = torch.zeros_like(labels, dtype=torch.bool)

        kept_embeddings = embeddings.clone().detach()  #.to(dtype=torch.float16)
        kept_labels = labels.clone().detach()

        if self._keep_only_good_samples:
            good_samples = torch.logical_not(batch_noisy_predictions)
            kept_embeddings = kept_embeddings[good_samples]
            kept_labels = kept_labels[good_samples]

        # Store current features
        self._memory_bank.add_to_memory(kept_embeddings, kept_labels)

        # Report if needed
        if self._inspector is not None:
            self._inspector.add_batch_info(batch_labels=labels,
                                           batch_predictions_scores=self._strategy.retrieve_batch_raw_scores(),
                                           batch_predictions_noisy=batch_noisy_predictions,
                                           batch_gt_noisy=self.current_batch_noisy_samples)

        return batch_noisy_predictions
