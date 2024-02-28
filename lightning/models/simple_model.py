import os
from typing import Type, Optional, Dict, Union, Literal, Tuple
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.functional import accuracy
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from lightning.data.dataset_utils import batch_unpack_function
from lightning.models.model_base import LitModelBase
from lightning.models.utils import construct_miner
from torch_metric_learning.noise_reducers import noise_reducers


class LitModel(LitModelBase):
    def __init__(self,
                 # Common model options
                 batch_size=256,
                 num_classes=10,
                 num_channels=1,
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
                 # Custom options
                 hidden_size=64,
                 width=28,
                 height=28
                 ):

        super().__init__(batch_size=batch_size,
                         num_classes=num_classes,
                         num_channels=num_channels,
                         use_pretrained_weights=use_pretrained_weights,
                         loss_class=loss_class,
                         loss_kwargs=loss_kwargs,
                         miner_class=miner_class,
                         miner_kwargs=miner_kwargs,
                         noise_reducer_class=noise_reducer_class,
                         noise_reducer_kwargs=noise_reducer_kwargs)
        self.save_hyperparameters()

    def _create_model(self) -> Tuple[torch.nn.Module, torch.nn.Module, int]:
        backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hparams.num_channels * self.hparams.width * self.hparams.height, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        classifier = nn.Linear(self.hparams.hidden_size, self.hparams.num_classes)
        return backbone, classifier, self.hparams.hidden_size


class SimpleMNISTModel(LightningModule):
    # Code moded from https://github.com/pytorch/examples/blob/main/mnist/main.py
    def __init__(self, learning_rate=1e-2):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128)
        )

        distance = distances.LpDistance(normalize_embeddings=True)
        reducer = reducers.ThresholdReducer(low=0)
        self.loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
        self.mining_func = miners.BatchHardMiner(distance=distance)
        self.batch_unpack_fn = batch_unpack_function
        # self.accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def forward_features(self, x):
        return self.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = self.batch_unpack_fn(batch, keys=('image', 'target'))

        embeddings = self(x)
        indices_tuple = self.mining_func(embeddings, y)
        loss = self.loss_func(embeddings, y, indices_tuple)

        self.log("train_loss_triplet", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_epoch_start(self):
        pass

    def training_epoch_end(self, training_step_outputs):
        pass
