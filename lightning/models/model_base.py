from typing import Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
import tqdm
from lightning_lite.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning import LightningModule
# from pytorch_lightning.core.decorators import auto_move_data
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR, CosineAnnealingLR
from torchmetrics.functional import accuracy
from warnings import warn
from copy import copy
import importlib

from lightning.data.dataset_utils import batch_unpack_function
from pytorch_metric_learning import losses, miners
from torch_metric_learning.noise_reducers import noise_reducers
from lightning.models.utils import *
#from utils import class_from_string


def class_from_string(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class DecoupledLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self._train_dataloader is None:
            self._train_dataloader = self.trainer.datamodule.train_dataloader()
        return self._train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self._val_dataloader is None:
            self._val_dataloader = self.trainer.datamodule.val_dataloader()
        return self._val_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self._test_dataloader is None:
            self._test_dataloader = self.trainer.datamodule.test_dataloader()
        return self._test_dataloader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()


# TODO: https://www.pytorchlightning.ai/blog/3-simple-tricks-that-will-change-the-way-you-debug-pytorch
class LitModelBase(DecoupledLightningModule):
    def __init__(self,
                 batch_size=256,
                 num_classes=1000,
                 num_channels=3,
                 use_pretrained_weights=True,
                 optimizer_class=torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 scheduler_class: Optional[Type[_LRScheduler]] = None,
                 scheduler_kwargs: Optional[Dict] = None,
                 loss_class: Union[Type[losses.BaseMetricLossFunction], Type[losses.BaseLossWrapper]] = losses.TripletMarginLoss,
                 loss_kwargs: Optional[Dict] = None,
                 miner_class: Type[miners.BaseMiner] = miners.BatchHardMiner,
                 miner_kwargs: Optional[Dict] = None,
                 noise_reducer_class: Type[noise_reducers.DefaultNoiseReducer] = noise_reducers.DefaultNoiseReducer,
                 noise_reducer_kwargs: Optional[Dict] = None,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone, self.linear_layer, self.embedding_size = \
            self._create_model()

        self.classification_loss = nn.CrossEntropyLoss()
        refined_loss_kwargs, cross_batch_memory_kwargs = self._populate_loss_args(loss_class, loss_kwargs)
        # refined_loss_kwargs['reducer'] = construct_reducer(reducer_class=reducers.AvgNonZeroReducer)

        self.miner = construct_miner(miner_class, self._populate_miner_args(miner_class, miner_kwargs), distance=None)

        if cross_batch_memory_kwargs is not None:
            metric_loss = loss_class(**refined_loss_kwargs)
            self.metric_loss = losses.CrossBatchMemory(metric_loss,
                                                       embedding_size=self.embedding_size,
                                                       memory_size=cross_batch_memory_kwargs['memory_size'],
                                                       miner=self.miner)
        else:
            self.metric_loss = loss_class(**refined_loss_kwargs)

        # TODO: Remove after inspection that is not needed / weighted was done within reducer instead
        # if type(self.miner).__name__ == 'PopulationAwareMiner' and \
        #         self.miner.pair_rejection_strategy.use_raw_probabilities:
        #     try:
        #         patch_object_with_distance(self.metric_loss)
        #     except Exception:
        #         warn('Unsupported distance patching !!!!!!!!!!')

        self.noise_reducer = \
            construct_noise_reducer(noise_reducer_class, noise_reducer_kwargs,
                                    embedding_size=self.embedding_size,
                                    num_classes=self.hparams.num_classes,
                                    cross_batch_memory_object=self.metric_loss
                                    if cross_batch_memory_kwargs is not None else None)

        # TODO: expose to config
        self.loss_weights = {'classification': 0.0, 'metric': 1.0}

        self.batch_unpack_fn = batch_unpack_function

    def _create_model(self) -> Tuple[torch.nn.Module, torch.nn.Module, int]:
        raise NotImplementedError

    def _populate_miner_args(self, miner_class, miner_kwargs):
        if miner_kwargs is not None:
            actual_miner_kwargs = copy(miner_kwargs)
        else:
            actual_miner_kwargs = {}

        init_args = get_init_args(miner_class)

        # No longer needed, however I leave it here for future reference
        if 'num_classes' in init_args:
            actual_miner_kwargs['num_classes'] = self.hparams.num_classes

        return actual_miner_kwargs

    def _populate_loss_args(self, loss_class, loss_kwargs) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Populates a dictionary which contains loss arguments. Some of them are given, some other are computed or bound
        to the data

        :param loss_class: Loss class
        :param loss_kwargs: Given keyword arguments
        :param num_classes: Number of classes
        :return:
        """
        # With specifying distance = None we let the default distance.
        # Some losses work only with specific distances and letting the user define it may be error-prone
        distance = None
        # Same with reducer
        reducer = None  # reducers.ThresholdReducer(low=0)

        # We get a copy of the loss args, because some of them are not for the loss specifically but for the optimizer
        # of the loss, e.g. in SoftTripleLoss. But also some
        if loss_kwargs is not None:
            actual_loss_kwargs = copy(loss_kwargs)
        else:
            actual_loss_kwargs = {}

        try:
            cross_batch_memory_kwargs = actual_loss_kwargs.pop('cross_batch_memory')
        except KeyError:
            cross_batch_memory_kwargs = None

        # Arguments required for some losses e.g., SoftTriple loss
        # Inspect the signature of the loss and parent classes in order to realize if these arguments are needed.
        # This is to avoid unexpected keyword argument errors
        init_args = get_init_args(loss_class)

        for key in list(actual_loss_kwargs.keys()):
            if key not in init_args:
                del actual_loss_kwargs[key]

        # in some losses these parameters are obtained from kwargs e.g. in
        # https://github.com/KevinMusgrave/pytorch-metric-learning/blob/10fd517a29d86d7752d0bdd7e0864377b89b3fc3
        # /src/pytorch_metric_learning/losses/subcenter_arcface_loss.py#L15
        if 'num_classes' in init_args or loss_class == losses.SubCenterArcFaceLoss:
            actual_loss_kwargs['num_classes'] = self.hparams.num_classes
        if 'embedding_size' in init_args or loss_class == losses.SubCenterArcFaceLoss:
            actual_loss_kwargs['embedding_size'] = self.embedding_size
        if 'distance' in init_args:
            actual_loss_kwargs['distance'] = distance
        if 'reducer' in init_args:
            actual_loss_kwargs['reducer'] = reducer

        if 'loss' in actual_loss_kwargs:
            # TODO: For now I leave it with no / default args
            actual_loss_kwargs['loss'] = class_from_string(actual_loss_kwargs['loss'])()

        return actual_loss_kwargs, cross_batch_memory_kwargs

    def forward(self, x, return_feat=False):
        feats = self.backbone(x)
        logits = self.linear_layer(feats)

        if return_feat:
            # TODO: Consider move of normalization outside of here, e.g. as in pytorch-metric-learning
            # feats = F.normalize(feats, p=2, dim=1)
            warn('Normalization is muted')
            return logits, feats
        else:
            return logits

    def forward_features(self, x):
        feats = self.backbone(x)
        return feats

    def _compute_metric_loss(self, feat, y, mined_indices):
        # Required step in case the miner has a distance weighting according to the mining logic
        if self.noise_reducer is not None and self.noise_reducer.strategy.use_raw_probabilities:
            # TODO: FIX THIS AWKWARD LOGIC
            try:
                # self.metric_loss.distance.weights = self.miner.pair_rejection_strategy.retrieve_batch_weights()
                self.metric_loss.reducer.weights = self.noise_reducer.strategy.retrieve_batch_weights().to('cuda:0')
                self.metric_loss.reducer.mined_indices = mined_indices
            except Exception:
                print('This metric loss does not support reducing')

        # Check if the loss is wrapped
        if isinstance(self.metric_loss, CrossBatchMemory) and self.noise_reducer is not None and \
                not self.noise_reducer.memory_is_dynamic():
            # The noise reducer has already enqueued the batch stuff in cross batch memory,
            # therefore we entirely omit enqueueing through the loss forward method
            return self.metric_loss(feat, y, enqueue_mask=torch.zeros_like(y, dtype=torch.bool))

        return self.metric_loss(feat, y, mined_indices)

    def _compute_classification_loss(self, logits, y):
        return self.classification_loss(logits, y)

    def _compute_loss(self, feat, logits, y, mined_tuples):
        metric_loss = self._compute_metric_loss(feat, y, mined_tuples)
        classification_loss = self._compute_classification_loss(logits, y)

        self.log("train_loss_metric", metric_loss)
        self.log("train_loss_classification", classification_loss)

        return self.loss_weights['classification'] * classification_loss + \
            self.loss_weights['metric'] * metric_loss

    def _remove_noisy_from_batch(self, batch, feat, y, logits):
        # Try to get the ground truth noisy information. This is used by the Dummy noise reducer for its reduction
        # and from other reducer for debugging / comparing against gt
        try:
            gt_noisy = self.batch_unpack_fn(batch, keys=('is_noisy',))[0]
        except (KeyError, TypeError):
            gt_noisy = None

        batch_noisy_predictions = self.noise_reducer(feat, y, noisy_samples=gt_noisy)

        batch_clean_predictions = torch.logical_not(batch_noisy_predictions)
        clean_feat = feat[batch_clean_predictions]
        clean_y = y[batch_clean_predictions]
        clean_logits = logits[batch_clean_predictions]

        return clean_feat, clean_y, clean_logits

    def training_step(self, batch, batch_idx):
        x, y = self.batch_unpack_fn(batch, keys=('image', 'target'))

        logits, feat = self(x, return_feat=True)

        if self.noise_reducer is not None:
            feat, y, logits = self._remove_noisy_from_batch(batch, feat, y, logits)

        indices_tuple = self.miner(feat, y)
        assert indices_tuple is not None
        total_loss = self._compute_loss(feat, logits, y, indices_tuple)
        return total_loss

    def _eval_step(self, batch, stage=None):
        x, y = self.batch_unpack_fn(batch, keys=('image', 'target'))
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self._compute_classification_loss(logits, y)
        acc = accuracy(preds, y, task='multiclass', num_classes=self.hparams.num_classes)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def on_fit_start(self):
        # If we have a logger and a noise reducer, patch the output path of noise reducer from logger's path
        if (self.logger is not None and
                self.noise_reducer is not None and self.noise_reducer.inspector is not None):
            logging_dir = self.logger.log_dir
            output_path = Path(logging_dir) / 'rejection_inspector_output'
            output_path.mkdir(exist_ok=True)
            self.noise_reducer.inspector._output_dir = output_path

    def on_train_start(self):
        # In this case we want the features from the pretrained model. We check the following cases:
        # (a) miner is PopulationAwareMiner, which is the only one supporting the option
        # (b) 'use_pretrained' is enabled
        # (c) model is loaded with pretrained weights
        if self.noise_reducer is not None and self.noise_reducer.use_pretrained and \
                self.hparams.use_pretrained_weights is True:
            embeddings_list = []
            labels_list = []
            print('Extracting features from pretrained model...')
            with torch.no_grad():
                train_dataloader = self.train_dataloader()
                for batch in tqdm.tqdm(train_dataloader):
                    x, y = self.batch_unpack_fn(batch, keys=('image', 'target'))
                    _, feat = self(x.to('cuda'), return_feat=True)
                    embeddings_list.append(feat.clone().detach())  # .to(dtype=torch.float16)
                    labels_list.append(y)

            all_features = torch.vstack(embeddings_list)
            all_class_labels = torch.hstack(labels_list)
            self.noise_reducer.bootstrap_initial(all_features, all_class_labels)

    def training_epoch_end(self, training_step_outputs):
        if self.noise_reducer is not None:
            self.noise_reducer.bootstrap_epoch(self.trainer.current_epoch)

    @property
    def num_training_steps(self) -> int:
        """
        Implementation from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449

        @return: Total training steps inferred from datamodule and devices.
        """
        warn(
            'The implementation may lack edge cases. See https://github.com/PyTorchLightning/pytorch-lightning/issues/5449')

        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        train_dataloader = self.train_dataloader()

        if hasattr(train_dataloader.sampler, 'n_iterations'):
            if train_dataloader.sampler.n_iterations > 0:
                return train_dataloader.sampler.n_iterations
            else:
                raise MisconfigurationException('Required steps are not provided by the sampler.')

        limit_batches = self.trainer.limit_train_batches
        batches = len(train_dataloader)
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def _construct_scheduler(self, optimizer, scheduler_class, scheduler_kwargs):
        """
        Some schedulers like OneCycleLR and CosineAnnealingLR require parameters that are not  known beforehand, but
        depend on runtime. E.g. number of steps can be obtained from the dataloader

        :param scheduler_class:
        :return:
        """

        # Regard None kwargs as empty dict
        scheduler_kwargs = {} if scheduler_kwargs is None else scheduler_kwargs

        # Construct a scheduler config, rather than returning a single scheduler in order to configure other params
        # such as interval etc
        scheduler_config = \
            {}

        # Special check for few schedulers
        if scheduler_class.__name__ == 'OneCycleLR':
            scheduler_config['interval'] = 'step'
            # Assign 0.1*lr as shown in OneCycleLR example, don't know why it is done like this
            # max_lr_list = [0.1 * pg['lr'] for pg in optimizer.param_groups]
            # Assign the lr itself as done in unicom
            max_lr_list = [pg['lr'] for pg in optimizer.param_groups]
            scheduler_config['scheduler'] = OneCycleLR(optimizer, max_lr_list, total_steps=self.num_training_steps)
        elif scheduler_class.__name__ == 'CosineAnnealingLR':
            scheduler_config['interval'] = 'step'
            scheduler_config['scheduler'] = CosineAnnealingLR(optimizer, T_max=self.num_training_steps)
        else:
            scheduler_config['scheduler'] = scheduler_class(**scheduler_kwargs)

        return scheduler_config

    def configure_optimizers(self):
        def ensure_type(class_type):
            if isinstance(class_type, str):
                module, class_name = class_type.rsplit('.', maxsplit=1)
                return getattr(importlib.import_module(module), class_name)
            else:
                return class_type

        optimizer_class = ensure_type(self.hparams.optimizer_class)

        # Construct the optimizer
        model_optimizer_kwargs = self.hparams.optimizer_kwargs

        # Look for existence of optimizer_kwargs in loss_args
        if 'optimizer_kwargs' not in self.hparams.loss_kwargs:
            optimizer = optimizer_class(self.backbone.parameters(), **model_optimizer_kwargs)
        else:
            loss_optimizer_kwargs = self.hparams.loss_kwargs['optimizer_kwargs']
            default_optimizer_kwargs = {}

            # We keep rest keys as default values in the optimizer
            for k, v in copy(model_optimizer_kwargs).items():
                if k not in loss_optimizer_kwargs:
                    default_optimizer_kwargs[k] = v
                    del model_optimizer_kwargs[k]
            param_groups = \
                [
                    {'params': self.backbone.parameters(), **model_optimizer_kwargs},
                    {'params': self.metric_loss.parameters(), **loss_optimizer_kwargs}
                ]
            optimizer = optimizer_class(param_groups, **default_optimizer_kwargs)

        optimizer_dict = {'optimizer': optimizer}

        if self.hparams.scheduler_class is not None:
            scheduler_class = ensure_type(self.hparams.scheduler_class)
            scheduler_kwargs = self.hparams.scheduler_kwargs

            optimizer_dict['lr_scheduler'] = self._construct_scheduler(optimizer, scheduler_class, scheduler_kwargs)

        return optimizer_dict
