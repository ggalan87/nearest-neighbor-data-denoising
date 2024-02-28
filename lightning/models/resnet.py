from typing import Optional, Type, Dict, Literal, Tuple, Union
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_metric_learning import losses, miners
from lightning.models.model_base import LitModelBase, Flatten
from lightning.models.backbones.blocks import DimReduceLayer
import lightning.models.backbones.preresnet
from torch_metric_learning.noise_reducers import noise_reducers
from .backbones.bninception import LinearNorm


class LitResnet(LitModelBase):
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
                 loss_class: Union[Type[losses.BaseMetricLossFunction], Type[losses.BaseLossWrapper]] = losses.TripletMarginLoss,
                 loss_kwargs: Optional[Dict] = None,
                 miner_class: Type[miners.BaseMiner] = miners.BatchHardMiner,
                 miner_kwargs: Optional[Dict] = None,
                 noise_reducer_class: Type[noise_reducers.DefaultNoiseReducer] = noise_reducers.DefaultNoiseReducer,
                 noise_reducer_kwargs: Optional[Dict] = None,
                 # Custom options
                 model_variant: Literal['resnet18', 'resnet50'] = 'resnet50',
                 dataset_variant: Literal['default', 'cifar', 'mnist'] = 'default',
                 reduced_dim: Optional[int] = 128,
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
        """

        @return: backbone and linear layer
        """
        # TODO: Use the provided function when upgrade torchvision
        # As of v0.14, TorchVision offers a new model registration mechanism which allows retreaving models and weights by
        # their names.

        if 'pre' in self.hparams.model_variant.lower():
            # model_class = getattr(lightning.models.backbones.preresnet, self.hparams.model_variant, None)
            raise NotImplementedError
        else:
            # Normal, torchvision models
            model_class = getattr(torchvision.models, self.hparams.model_variant, None)

        if model_class is None:
            raise AssertionError(f'Invalid model name {self.hparams.model_variant}')

        if self.hparams.dataset_variant != 'default' and self.hparams.use_pretrained_weights is True:
            raise AssertionError('Cannot use pretrained weights due to model pruning')

        if self.hparams.dataset_variant == 'default':
            # For now I always use the default weights.
            weights = "DEFAULT" if self.hparams.use_pretrained_weights else None
            model = model_class(weights=weights)
        elif self.hparams.dataset_variant == 'cifar':
            # Changes below were proposed for CIFAR10 in pytorch lightning examples
            model = model_class(weights=None, num_classes=self.hparams.num_classes)
            model.conv1 = nn.Conv2d(self.hparams.num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                    bias=False)
            model.maxpool = nn.Identity()
        elif self.hparams.dataset_variant == 'mnist':
            model = model_class(weights=None, num_classes=self.hparams.num_classes)
            model.conv1 = nn.Conv2d(self.hparams.num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                    bias=False)
        else:
            raise NotImplementedError(self.hparams.dataset_variant)

        # disable the internal classifier and define ours outside the module
        model.fc = nn.Identity()

        embedding_size = 512 * model.layer1[0].__class__.expansion

        if self.hparams.reduced_dim is not None:
            dropout = nn.Dropout(p=0.5)
            # reduce_layer = (
            #     nn.Sequential(
            #         DimReduceLayer(embedding_size, self.hparams.reduced_dim, nonlinear='relu'),
            #         nn.AdaptiveAvgPool2d((1, 1)),
            #     ))
            reduce_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                LinearNorm(embedding_size, self.hparams.reduced_dim)
            )

            model.avgpool = nn.Identity()

            model = \
                nn.Sequential(
                    model,
                    nn.Unflatten(1, (embedding_size, 7, 7)),
                    dropout,
                    reduce_layer,
                    nn.Flatten()
                )

            embedding_size = self.hparams.reduced_dim

        fc = \
            nn.Sequential(
                Flatten(),
                nn.Linear(embedding_size, self.hparams.num_classes)
            )

        return model, fc, embedding_size


def check_model():
    optimizer_class = torch.optim.Adam
    optimizer_kwargs = \
        {
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'eps': 0.01
        }

    scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_kwargs = \
        {
            'step_size': 20
        }
    loss_class = losses.SoftTripleLoss
    loss_kwargs = \
        {
            'lr': 1e-2
        }
    resnet_model = LitResnet(model_variant='resnet50', num_classes=98,
                             optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs,
                             scheduler_class=scheduler_class, scheduler_kwargs=scheduler_kwargs,
                             loss_class=loss_class, loss_kwargs=loss_kwargs, reduced_dim=128)


if __name__ == '__main__':
    check_model()
