from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/bninception.py
from torch.nn import Sequential


class BNInception(nn.Module):
    def __init__(self, **kwargs):
        super(BNInception, self).__init__()
        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
        )
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d(
            (3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True
        )
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(
            64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv2_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d(
            (3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True
        )
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(
            192, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3a_3x3_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(
            192, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(
            96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_3a_pool_proj = nn.Conv2d(
            192, 32, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3b_3x3_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(
            96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_3b_pool_proj = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(
            320, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(
            128, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.inception_3c_3x3_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(
            320, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(
            96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d(
            (3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True
        )
        self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(
            576, 64, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4a_3x3_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(
            576, 96, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(
            96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_4a_pool_proj = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(
            576, 96, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(
            96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4b_3x3_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(
            576, 96, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(
            96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_4b_pool_proj = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(
            128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4c_3x3_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(
            128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(
            160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_4c_pool_proj = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(
            608, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(
            128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4d_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(
            608, 160, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(
            160, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_4d_pool_proj = nn.Conv2d(
            608, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(
            608, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(
            128, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.inception_4e_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(
            608, 192, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(
            192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d(
            (3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True
        )
        self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(352, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(
            1056, 192, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(
            192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5a_3x3_bn = nn.BatchNorm2d(320, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(
            1056, 160, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(
            160, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(
            224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True
        )
        self.inception_5a_pool_proj = nn.Conv2d(
            1056, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(352, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(
            1024, 192, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(
            192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5b_3x3_bn = nn.BatchNorm2d(320, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(
            1024, 192, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(
            192, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(
            224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d(
            (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), ceil_mode=True
        )
        self.inception_5b_pool_proj = nn.Conv2d(
            1024, 128, kernel_size=(1, 1), stride=(1, 1)
        )
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)

    def features(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_relu_7x7_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_relu_3x3_reduce_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_relu_3x3_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(
            inception_3a_3x3_reduce_out
        )
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(
            inception_3a_3x3_reduce_bn_out
        )
        inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_relu_3x3_reduce_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(
            pool2_3x3_s2_out
        )
        inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(
            inception_3a_double_3x3_reduce_out
        )
        inception_3a_relu_double_3x3_reduce_out = self.inception_3a_relu_double_3x3_reduce(
            inception_3a_double_3x3_reduce_bn_out
        )
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(
            inception_3a_relu_double_3x3_reduce_out
        )
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(
            inception_3a_double_3x3_1_out
        )
        inception_3a_relu_double_3x3_1_out = self.inception_3a_relu_double_3x3_1(
            inception_3a_double_3x3_1_bn_out
        )
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(
            inception_3a_relu_double_3x3_1_out
        )
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(
            inception_3a_double_3x3_2_out
        )
        inception_3a_relu_double_3x3_2_out = self.inception_3a_relu_double_3x3_2(
            inception_3a_double_3x3_2_bn_out
        )
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(
            inception_3a_pool_proj_out
        )
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(
            inception_3a_pool_proj_bn_out
        )
        inception_3a_output_out = torch.cat(
            [
                inception_3a_relu_1x1_out,
                inception_3a_relu_3x3_out,
                inception_3a_relu_double_3x3_2_out,
                inception_3a_relu_pool_proj_out,
            ],
            1,
        )
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(
            inception_3a_output_out
        )
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(
            inception_3b_3x3_reduce_out
        )
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(
            inception_3b_3x3_reduce_bn_out
        )
        inception_3b_3x3_out = self.inception_3b_3x3(inception_3b_relu_3x3_reduce_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(
            inception_3a_output_out
        )
        inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(
            inception_3b_double_3x3_reduce_out
        )
        inception_3b_relu_double_3x3_reduce_out = self.inception_3b_relu_double_3x3_reduce(
            inception_3b_double_3x3_reduce_bn_out
        )
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(
            inception_3b_relu_double_3x3_reduce_out
        )
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(
            inception_3b_double_3x3_1_out
        )
        inception_3b_relu_double_3x3_1_out = self.inception_3b_relu_double_3x3_1(
            inception_3b_double_3x3_1_bn_out
        )
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(
            inception_3b_relu_double_3x3_1_out
        )
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(
            inception_3b_double_3x3_2_out
        )
        inception_3b_relu_double_3x3_2_out = self.inception_3b_relu_double_3x3_2(
            inception_3b_double_3x3_2_bn_out
        )
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(
            inception_3b_pool_proj_out
        )
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(
            inception_3b_pool_proj_bn_out
        )
        inception_3b_output_out = torch.cat(
            [
                inception_3b_relu_1x1_out,
                inception_3b_relu_3x3_out,
                inception_3b_relu_double_3x3_2_out,
                inception_3b_relu_pool_proj_out,
            ],
            1,
        )
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(
            inception_3b_output_out
        )
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(
            inception_3c_3x3_reduce_out
        )
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(
            inception_3c_3x3_reduce_bn_out
        )
        inception_3c_3x3_out = self.inception_3c_3x3(inception_3c_relu_3x3_reduce_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(
            inception_3b_output_out
        )
        inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(
            inception_3c_double_3x3_reduce_out
        )
        inception_3c_relu_double_3x3_reduce_out = self.inception_3c_relu_double_3x3_reduce(
            inception_3c_double_3x3_reduce_bn_out
        )
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(
            inception_3c_relu_double_3x3_reduce_out
        )
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(
            inception_3c_double_3x3_1_out
        )
        inception_3c_relu_double_3x3_1_out = self.inception_3c_relu_double_3x3_1(
            inception_3c_double_3x3_1_bn_out
        )
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(
            inception_3c_relu_double_3x3_1_out
        )
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(
            inception_3c_double_3x3_2_out
        )
        inception_3c_relu_double_3x3_2_out = self.inception_3c_relu_double_3x3_2(
            inception_3c_double_3x3_2_bn_out
        )
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat(
            [
                inception_3c_relu_3x3_out,
                inception_3c_relu_double_3x3_2_out,
                inception_3c_pool_out,
            ],
            1,
        )
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(
            inception_3c_output_out
        )
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(
            inception_4a_3x3_reduce_out
        )
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(
            inception_4a_3x3_reduce_bn_out
        )
        inception_4a_3x3_out = self.inception_4a_3x3(inception_4a_relu_3x3_reduce_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = self.inception_4a_double_3x3_reduce(
            inception_3c_output_out
        )
        inception_4a_double_3x3_reduce_bn_out = self.inception_4a_double_3x3_reduce_bn(
            inception_4a_double_3x3_reduce_out
        )
        inception_4a_relu_double_3x3_reduce_out = self.inception_4a_relu_double_3x3_reduce(
            inception_4a_double_3x3_reduce_bn_out
        )
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(
            inception_4a_relu_double_3x3_reduce_out
        )
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(
            inception_4a_double_3x3_1_out
        )
        inception_4a_relu_double_3x3_1_out = self.inception_4a_relu_double_3x3_1(
            inception_4a_double_3x3_1_bn_out
        )
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(
            inception_4a_relu_double_3x3_1_out
        )
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(
            inception_4a_double_3x3_2_out
        )
        inception_4a_relu_double_3x3_2_out = self.inception_4a_relu_double_3x3_2(
            inception_4a_double_3x3_2_bn_out
        )
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(
            inception_4a_pool_proj_out
        )
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(
            inception_4a_pool_proj_bn_out
        )
        inception_4a_output_out = torch.cat(
            [
                inception_4a_relu_1x1_out,
                inception_4a_relu_3x3_out,
                inception_4a_relu_double_3x3_2_out,
                inception_4a_relu_pool_proj_out,
            ],
            1,
        )
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(
            inception_4a_output_out
        )
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(
            inception_4b_3x3_reduce_out
        )
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(
            inception_4b_3x3_reduce_bn_out
        )
        inception_4b_3x3_out = self.inception_4b_3x3(inception_4b_relu_3x3_reduce_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(
            inception_4a_output_out
        )
        inception_4b_double_3x3_reduce_bn_out = self.inception_4b_double_3x3_reduce_bn(
            inception_4b_double_3x3_reduce_out
        )
        inception_4b_relu_double_3x3_reduce_out = self.inception_4b_relu_double_3x3_reduce(
            inception_4b_double_3x3_reduce_bn_out
        )
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(
            inception_4b_relu_double_3x3_reduce_out
        )
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(
            inception_4b_double_3x3_1_out
        )
        inception_4b_relu_double_3x3_1_out = self.inception_4b_relu_double_3x3_1(
            inception_4b_double_3x3_1_bn_out
        )
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(
            inception_4b_relu_double_3x3_1_out
        )
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(
            inception_4b_double_3x3_2_out
        )
        inception_4b_relu_double_3x3_2_out = self.inception_4b_relu_double_3x3_2(
            inception_4b_double_3x3_2_bn_out
        )
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(
            inception_4b_pool_proj_out
        )
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(
            inception_4b_pool_proj_bn_out
        )
        inception_4b_output_out = torch.cat(
            [
                inception_4b_relu_1x1_out,
                inception_4b_relu_3x3_out,
                inception_4b_relu_double_3x3_2_out,
                inception_4b_relu_pool_proj_out,
            ],
            1,
        )
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(
            inception_4b_output_out
        )
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(
            inception_4c_3x3_reduce_out
        )
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(
            inception_4c_3x3_reduce_bn_out
        )
        inception_4c_3x3_out = self.inception_4c_3x3(inception_4c_relu_3x3_reduce_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = self.inception_4c_double_3x3_reduce(
            inception_4b_output_out
        )
        inception_4c_double_3x3_reduce_bn_out = self.inception_4c_double_3x3_reduce_bn(
            inception_4c_double_3x3_reduce_out
        )
        inception_4c_relu_double_3x3_reduce_out = self.inception_4c_relu_double_3x3_reduce(
            inception_4c_double_3x3_reduce_bn_out
        )
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(
            inception_4c_relu_double_3x3_reduce_out
        )
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(
            inception_4c_double_3x3_1_out
        )
        inception_4c_relu_double_3x3_1_out = self.inception_4c_relu_double_3x3_1(
            inception_4c_double_3x3_1_bn_out
        )
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(
            inception_4c_relu_double_3x3_1_out
        )
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(
            inception_4c_double_3x3_2_out
        )
        inception_4c_relu_double_3x3_2_out = self.inception_4c_relu_double_3x3_2(
            inception_4c_double_3x3_2_bn_out
        )
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(
            inception_4c_pool_proj_out
        )
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(
            inception_4c_pool_proj_bn_out
        )
        inception_4c_output_out = torch.cat(
            [
                inception_4c_relu_1x1_out,
                inception_4c_relu_3x3_out,
                inception_4c_relu_double_3x3_2_out,
                inception_4c_relu_pool_proj_out,
            ],
            1,
        )
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(
            inception_4c_output_out
        )
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(
            inception_4d_3x3_reduce_out
        )
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(
            inception_4d_3x3_reduce_bn_out
        )
        inception_4d_3x3_out = self.inception_4d_3x3(inception_4d_relu_3x3_reduce_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = self.inception_4d_double_3x3_reduce(
            inception_4c_output_out
        )
        inception_4d_double_3x3_reduce_bn_out = self.inception_4d_double_3x3_reduce_bn(
            inception_4d_double_3x3_reduce_out
        )
        inception_4d_relu_double_3x3_reduce_out = self.inception_4d_relu_double_3x3_reduce(
            inception_4d_double_3x3_reduce_bn_out
        )
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(
            inception_4d_relu_double_3x3_reduce_out
        )
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(
            inception_4d_double_3x3_1_out
        )
        inception_4d_relu_double_3x3_1_out = self.inception_4d_relu_double_3x3_1(
            inception_4d_double_3x3_1_bn_out
        )
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(
            inception_4d_relu_double_3x3_1_out
        )
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(
            inception_4d_double_3x3_2_out
        )
        inception_4d_relu_double_3x3_2_out = self.inception_4d_relu_double_3x3_2(
            inception_4d_double_3x3_2_bn_out
        )
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(
            inception_4d_pool_proj_out
        )
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(
            inception_4d_pool_proj_bn_out
        )
        inception_4d_output_out = torch.cat(
            [
                inception_4d_relu_1x1_out,
                inception_4d_relu_3x3_out,
                inception_4d_relu_double_3x3_2_out,
                inception_4d_relu_pool_proj_out,
            ],
            1,
        )
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(
            inception_4d_output_out
        )
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(
            inception_4e_3x3_reduce_out
        )
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(
            inception_4e_3x3_reduce_bn_out
        )
        inception_4e_3x3_out = self.inception_4e_3x3(inception_4e_relu_3x3_reduce_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(
            inception_4d_output_out
        )
        inception_4e_double_3x3_reduce_bn_out = self.inception_4e_double_3x3_reduce_bn(
            inception_4e_double_3x3_reduce_out
        )
        inception_4e_relu_double_3x3_reduce_out = self.inception_4e_relu_double_3x3_reduce(
            inception_4e_double_3x3_reduce_bn_out
        )
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(
            inception_4e_relu_double_3x3_reduce_out
        )
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(
            inception_4e_double_3x3_1_out
        )
        inception_4e_relu_double_3x3_1_out = self.inception_4e_relu_double_3x3_1(
            inception_4e_double_3x3_1_bn_out
        )
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(
            inception_4e_relu_double_3x3_1_out
        )
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(
            inception_4e_double_3x3_2_out
        )
        inception_4e_relu_double_3x3_2_out = self.inception_4e_relu_double_3x3_2(
            inception_4e_double_3x3_2_bn_out
        )
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat(
            [
                inception_4e_relu_3x3_out,
                inception_4e_relu_double_3x3_2_out,
                inception_4e_pool_out,
            ],
            1,
        )
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(
            inception_4e_output_out
        )
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(
            inception_5a_3x3_reduce_out
        )
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(
            inception_5a_3x3_reduce_bn_out
        )
        inception_5a_3x3_out = self.inception_5a_3x3(inception_5a_relu_3x3_reduce_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(
            inception_4e_output_out
        )
        inception_5a_double_3x3_reduce_bn_out = self.inception_5a_double_3x3_reduce_bn(
            inception_5a_double_3x3_reduce_out
        )
        inception_5a_relu_double_3x3_reduce_out = self.inception_5a_relu_double_3x3_reduce(
            inception_5a_double_3x3_reduce_bn_out
        )
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(
            inception_5a_relu_double_3x3_reduce_out
        )
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(
            inception_5a_double_3x3_1_out
        )
        inception_5a_relu_double_3x3_1_out = self.inception_5a_relu_double_3x3_1(
            inception_5a_double_3x3_1_bn_out
        )
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(
            inception_5a_relu_double_3x3_1_out
        )
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(
            inception_5a_double_3x3_2_out
        )
        inception_5a_relu_double_3x3_2_out = self.inception_5a_relu_double_3x3_2(
            inception_5a_double_3x3_2_bn_out
        )
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(
            inception_5a_pool_proj_out
        )
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(
            inception_5a_pool_proj_bn_out
        )
        inception_5a_output_out = torch.cat(
            [
                inception_5a_relu_1x1_out,
                inception_5a_relu_3x3_out,
                inception_5a_relu_double_3x3_2_out,
                inception_5a_relu_pool_proj_out,
            ],
            1,
        )
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(
            inception_5a_output_out
        )
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(
            inception_5b_3x3_reduce_out
        )
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(
            inception_5b_3x3_reduce_bn_out
        )
        inception_5b_3x3_out = self.inception_5b_3x3(inception_5b_relu_3x3_reduce_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(
            inception_5a_output_out
        )
        inception_5b_double_3x3_reduce_bn_out = self.inception_5b_double_3x3_reduce_bn(
            inception_5b_double_3x3_reduce_out
        )
        inception_5b_relu_double_3x3_reduce_out = self.inception_5b_relu_double_3x3_reduce(
            inception_5b_double_3x3_reduce_bn_out
        )
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(
            inception_5b_relu_double_3x3_reduce_out
        )
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(
            inception_5b_double_3x3_1_out
        )
        inception_5b_relu_double_3x3_1_out = self.inception_5b_relu_double_3x3_1(
            inception_5b_double_3x3_1_bn_out
        )
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(
            inception_5b_relu_double_3x3_1_out
        )
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(
            inception_5b_double_3x3_2_out
        )
        inception_5b_relu_double_3x3_2_out = self.inception_5b_relu_double_3x3_2(
            inception_5b_double_3x3_2_bn_out
        )
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(
            inception_5b_pool_proj_out
        )
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(
            inception_5b_pool_proj_bn_out
        )
        inception_5b_output_out = torch.cat(
            [
                inception_5b_relu_1x1_out,
                inception_5b_relu_3x3_out,
                inception_5b_relu_double_3x3_2_out,
                inception_5b_relu_pool_proj_out,
            ],
            1,
        )
        return inception_5b_output_out

    def logits(self, features):
        x = F.adaptive_max_pool2d(features, output_size=1)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if "last_linear" in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class LinearNorm(nn.Module):
    def __init__(self, in_channels, head_dim):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_channels, head_dim)
        self.fc.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def build_backbone(freeze_bn=False):
    model = BNInception()

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    return model


def build_head(with_dropout=False):
    if with_dropout:
        # return nn.Sequential(LinearNorm(1024, 512), nn.ReLU(), nn.Dropout(p=0.2))
        return nn.Sequential(LinearNorm(1024, 512), nn.Dropout(p=0.2))
    else:
        return LinearNorm(1024, 512)


def build_model(freeze_bn=False, with_dropout=False):
    backbone = build_backbone(freeze_bn)
    backbone.load_param(os.path.expanduser("~/.torch/models/bn_inception-52deb4733.pth"))

    head = build_head(with_dropout=with_dropout)

    model = Sequential(OrderedDict([("backbone", backbone), ("head", head)]))
    return model
