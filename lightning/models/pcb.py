import torch
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import functional as F
from lightning.models.model_base import DecoupledLightningModule

from typing import Optional, Type, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torchvision.models.resnet import BasicBlock
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.core.decorators import auto_move_data
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from warnings import warn
import psutil
from .backbones.blocks import DimReduceLayer
# from losses import TripletLoss
from lightning.losses_playground import PopulationAwareTripletLoss
from lightning.data.dataset_utils import batch_unpack_function

__all__ = ['pcb_p6', 'pcb_p4']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PCB(nn.Module):
    """Part-based Convolutional Baseline.
    Reference:
        Sun et al. Beyond Part Models: Person Retrieval with Refined
        Part Pooling (and A Strong Convolutional Baseline). ECCV 2018.
    Public keys:
        - ``pcb_p4``: PCB with 4-part strips.
        - ``pcb_p6``: PCB with 6-part strips.
    """

    def __init__(
            self,
            num_classes,
            loss,
            block,
            layers,
            parts=6,
            reduced_dim=256,
            nonlinear='relu',
            **kwargs
    ):
        self.inplanes = 64
        super(PCB, self).__init__()
        self.loss = loss
        self.parts = parts
        self.feature_dim = 512 * block.expansion
        self.use_reduced = False

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # pcb layers
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = DimReduceLayer(
            512 * block.expansion, reduced_dim, nonlinear=nonlinear
        )
        self.feature_dim = reduced_dim
        self.classifier = nn.ModuleList(
            [
                nn.Linear(self.feature_dim, num_classes)
                for _ in range(self.parts)
            ]
        )

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v_g = self.parts_avgpool(f)

        if not self.training:
            if not self.use_reduced:
                v_g = F.normalize(v_g, p=2, dim=1)
                return v_g.view(v_g.size(0), -1)
            else:
                # no dropout during testing
                v_h = self.conv5(v_g)
                feats = []
                for i in range(self.parts):
                    v_h_i = v_h[:, :, i, :]
                    v_h_i = v_h_i.view(v_h_i.size(0), -1)
                    feats.append(v_h_i)
                feat = torch.cat(tuple(feats), 1)
                feat = F.normalize(feat, p=2, dim=1)
                return feat

        v_g = self.dropout(v_g)
        v_h = self.conv5(v_g)

        y = []
        for i in range(self.parts):
            v_h_i = v_h[:, :, i, :]
            v_h_i = v_h_i.view(v_h_i.size(0), -1)
            y_i = self.classifier[i](v_h_i)
            y.append(y_i)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            # v_g = F.normalize(v_g, p=2, dim=1)
            # return y, v_g.view(v_g.size(0), -1)
            v_h_n = F.normalize(v_h, p=2, dim=1)
            return y, v_h_n.view(v_h_n.size(0), -1)
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def pcb_p6(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=6,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def pcb_p4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=4,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def DeepSupervision(criterion, xs, y):
    """DeepSupervision
    Applies criterion to each element in a list.
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss


class LitPCB(DecoupledLightningModule):
    def __init__(self, lr=0.1, batch_size=64, num_classes=10, pretrained=True, batch_unpack_fn=None,
                 loss_class: Optional[Type] = None, loss_args: Optional[Dict] = None):
        super().__init__()

        self.save_hyperparameters()
        self.model = pcb_p6(num_classes=num_classes, loss='triplet', pretrained=pretrained)

        loss_args = loss_args if loss_args is not None else {}

        # Default to triplet loss if not explicitly specified
        if loss_class is None:
            self.triplet_loss = TripletLoss(**loss_args)
        else:
            self.triplet_loss = loss_class(**loss_args)

        if batch_unpack_fn is None:
            self.batch_unpack_fn = batch_unpack_function
        else:
            self.batch_unpack_fn = batch_unpack_fn

    def forward(self, x, return_feat=False):
        logits, feats = self.model.forward(x)

        if return_feat:
            return logits, feats
        else:
            return logits

    def forward_features(self, x):
        if self.model.training:
            warn('Called forward feats while in training. Is this correct?')
            _, feats = self.model.forward(x)
        else:
            feats = self.model.forward(x)
        return feats

    def compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = self.batch_unpack_fn(batch, keys=('image', 'id'))
        logits, feat = self(x, return_feat=True)
        loss = self.compute_loss(F.cross_entropy, logits, y)

        #tloss = self.triplet_loss(feat.view(feat.size(0), -1), y)
        tloss = self.compute_loss(self.triplet_loss, feat, y)
        total_loss = loss + tloss

        self.log("train_loss_cross_entropy", loss)
        self.log("train_loss_triplet", tloss)
        self.log("train_loss_total", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError('Cannot implement validation step for identity models')

    def test_step(self, batch, batch_idx):
        raise NotImplementedError('Cannot implement test step for identity models')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                total_steps=self.num_training_steps
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    @property
    def num_training_steps(self) -> int:
        """
        Implementation from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449

        @return: The number of steps per epoch
        """
        warn(
            'The implementation may lack edge cases. See https://github.com/PyTorchLightning/pytorch-lightning/issues/5449')

        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def training_epoch_end(self, training_step_outputs):
        # TODO: implement fixed layers for few epochs as in torchreid here or in separate base class
        #  https://github.com/KaiyangZhou/deep-person-reid/blob/e34e3ae85fe02314e3a9a9a93c5828f4fed1b225/torchreid/engine/engine.py#L456

        if isinstance(self.triplet_loss, PopulationAwareTripletLoss):
            self.triplet_loss.bootstrap_epoch(self.trainer.current_epoch)
            self.triplet_loss.ei.report()
            self.triplet_loss.ei.reset_store()