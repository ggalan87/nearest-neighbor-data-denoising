from typing import Type, Dict, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchmetrics.functional import accuracy
from lightning.data.dataset_utils import batch_unpack_function
from pytorch_metric_learning import distances, losses, miners, reducers
from lightning.models.utils import construct_miner
from lightning.models.model_base import DecoupledLightningModule, Flatten


class ResNetMNIST(DecoupledLightningModule):
    def __init__(self,
                 num_classes=10,
                 channels=1,
                 loss_class: Type[losses.BaseMetricLossFunction] = losses.TripletMarginLoss,
                 loss_kwargs: Optional[Dict] = None,
                 miner_class: Type[miners.BaseMiner] = miners.BatchHardMiner,
                 miner_kwargs: Optional[Dict] = None,
                 noise_reducer: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = torchvision.models.resnet18(num_classes=num_classes)
        self.backbone.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # disable the internal classifier and define ours outside the module
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(512 * self.backbone.layer1[0].__class__.expansion, num_classes)
        )

        # self.loss = nn.CrossEntropyLoss()

        # With specifying distance = None we let the default distance.
        # Some losses work only with specific distances and letting the user define it may be error-prone
        distance = None

        self.miner_class = miner_class

        self.noise_reducer = noise_reducer

        self.miner = construct_miner(noise_reducer, miner_class, miner_kwargs, distance)

        reducer = reducers.ThresholdReducer(low=0)

        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss = loss_class(**loss_kwargs, distance=distance, reducer=reducer)

        self.batch_unpack_fn = batch_unpack_function

    def forward(self, x, return_feat=False):
        feats = self.backbone(x)
        logits = self.classifier(feats)

        if return_feat:
            return logits, feats
        else:
            return logits

    def forward_features(self, x):
        feat = self.backbone(x)
        return feat

    def training_step(self, batch, batch_no):
        x, y = self.batch_unpack_fn(batch, keys=('image', 'target', ))

        try:
            noisy = self.batch_unpack_fn(batch, keys=('noisy', ))[0]
        except KeyError:
            noisy = None

        logits, feat = self(x, return_feat=True)

        # loss = self.loss(logits, y)

        if noisy is not None and self.with_population_aware:
            indices_tuple = self.miner(feat, y, noisy_samples=noisy)
        else:
            indices_tuple = self.miner(feat, y)

        loss = self.loss(feat, y, indices_tuple)

        return loss

    def evaluate(self, batch, stage=None):
        x, y = self.batch_unpack_fn(batch, keys=('image', 'target'))
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_epoch_end(self, training_step_outputs):
        # if isinstance(mining_func, PopulationAwareMiner):
        if type(self.miner).__name__ == 'PopulationAwareMiner':
            self.miner.bootstrap_epoch(self.trainer.current_epoch)
            self.miner.ei.report()
            self.miner.ei.reset_store()
