from typing import Optional, Type, Dict, Literal, Tuple, Union
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_metric_learning import losses, miners
from lightning.models.model_base import LitModelBase, Flatten
from warnings import warn
from torchmetrics.functional import accuracy
from lightning.models.backbones.lenet import LeNet, LeNet_plus_plus


class LitLenet(LitModelBase):
    def __init__(self,
                 # Common model options
                 batch_size=256,
                 num_classes=1000,
                 num_channels=3,
                 use_pretrained_weights=True,
                 optimizer_class=torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 scheduler_class: Optional[Type[_LRScheduler]] = None,
                 scheduler_kwargs: Optional[Dict] = None,
                 loss_class: Union[
                     Type[losses.BaseMetricLossFunction], Type[losses.BaseLossWrapper]] = losses.TripletMarginLoss,
                 loss_kwargs: Optional[Dict] = None,
                 miner_class: Type[miners.BaseMiner] = miners.BatchHardMiner,
                 miner_kwargs: Optional[Dict] = None,
                 noise_reducer: Optional[str] = None,
                 noise_reducer_kwargs: Optional[Dict] = None,
                 # Custom options
                 model_variant: Literal['lenet', 'lenet++'] = 'lenet++'
                 ):
        super().__init__(batch_size=batch_size,
                         num_classes=num_classes,
                         num_channels=num_channels,
                         use_pretrained_weights=use_pretrained_weights,
                         loss_class=loss_class,
                         loss_kwargs=loss_kwargs,
                         miner_class=miner_class,
                         miner_kwargs=miner_kwargs,
                         noise_reducer=noise_reducer,
                         noise_reducer_kwargs=noise_reducer_kwargs)

        self.save_hyperparameters()
