from typing import Optional, Type, Dict, Literal, Tuple, Union
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_metric_learning import losses, miners
from lightning.models.model_base import LitModelBase, Flatten
from warnings import warn
from torchmetrics.functional import accuracy

from torch_metric_learning.noise_reducers import noise_reducers
from .backbones.bninception import build_model as bninception_build_model


class LitInception(LitModelBase):
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
                 model_variant: Literal['inceptionv3', 'bninception'] = 'inceptionv3',
                 freeze_bn: bool = False,
                 with_dropout=False
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

        if self.hparams.model_variant == 'inceptionv3':
            weights = "DEFAULT" if self.hparams.use_pretrained_weights else None
            model = torchvision.models.inception_v3(weights=weights)

            # disable the internal classifier and define ours outside the module
            # Inception v3 has another auxiliary classifier, which we leave it as s, because we don't use the features  prior
            # to it
            model.fc = nn.Identity()

            fc = \
                nn.Sequential(
                    Flatten(),
                    nn.Linear(2048, self.hparams.num_classes)
                )

            return model, fc, 2048

        elif self.hparams.model_variant == 'bninception':
            model = bninception_build_model(self.hparams.freeze_bn, self.hparams.with_dropout)

            fc = \
                nn.Sequential(
                    Flatten(),
                    nn.Linear(512, self.hparams.num_classes)
                )

            return model, fc, 512

        else:
            raise NotImplementedError

    def forward(self, x, return_feat=False):
        inception_outputs = self.backbone(x)
        if isinstance(inception_outputs, torchvision.models.InceptionOutputs):
            feats, aux_logits = inception_outputs
        else:
            feats = inception_outputs
            aux_logits = None

        logits = self.linear_layer(feats)

        logits_output = logits if aux_logits is None else (logits, aux_logits)

        if return_feat:
            # TODO: Consider move of normalization outside of here, e.g. as in pytorch-metric-learning
            # feats = F.normalize(feats, p=2, dim=1)
            warn('Normalization is muted')
            return logits_output, feats
        else:
            return logits_output

    def _compute_classification_loss(self, logits, y):
        if isinstance(logits, tuple):
            logits, aux_logits = logits
            # Weighted criterion from:
            # https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            return self.classification_loss(logits, y) + 0.4 * self.classification_loss(aux_logits, y)
        else:
            return self.classification_loss(logits, y)

    def _check_input_size(self):
        pass

    def on_fit_start(self):
        super().on_fit_start()

        self._check_input_size()

    def on_test_start(self):
        self._check_input_size()


if __name__ == '__main__':
    inception_model = LitInception()
    inputs = torch.rand(64, 3, 299, 299)
    inception_model(inputs)
    pass