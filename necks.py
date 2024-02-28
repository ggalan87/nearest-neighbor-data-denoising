import torch
import torch.nn as nn

from mmcls.models.builder import NECKS
from mmcls.models.necks import GlobalAveragePooling

class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

@NECKS.register_module()
class DimNeck(nn.Module):
    """Global Average Pooling neck.
    Note that we use `view` to remove extra channel after pooling.
    We do not use `squeeze` as it will also remove the batch dimension
    when the tensor has a batch dimension of size 1, which can lead to
    unexpected errors.
    """

    def __init__(self, out_channels=512):
        super(DimNeck, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)

        self.conv5 = DimReduceLayer(
            in_channels=2048, out_channels=out_channels, nonlinear='relu'
        )

    def init_weights(self):
        pass

    def forward(self, inputs):
        # if isinstance(inputs, tuple):
        #     outs = tuple([self.gap(x) for x in inputs])
        #     outs = tuple(
        #         [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        # elif isinstance(inputs, torch.Tensor):
        outs = self.gap(inputs)
        outs = self.dropout(outs)
        outs = self.conv5(outs)
        outs = outs.view(inputs.size(0), -1)
        # else:
        #     raise TypeError('neck inputs should be tuple or torch.tensor')

        return outs