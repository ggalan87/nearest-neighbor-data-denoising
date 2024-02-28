import math
import torch
from pytorch_metric_learning.losses.arcface_loss import ArcFaceLoss
from torch_metric_learning.losses.combined_margin_loss import CombinedMarginLoss


def compare():
    # Parameters
    scale = 32
    margin = 0.3

    num_classes = 100
    embedding_size = 128
    batch_size = 64

    inputs = torch.rand((batch_size, embedding_size))

    unicom_loss = CombinedMarginLoss(
        s=scale,
        m1=1.0,
        m2=margin,
        m3=0.0,
        interclass_filtering_threshold=0.0
    )

    # The implementation below gets margin in degrees
    arcface_loss = ArcFaceLoss(num_classes=num_classes,
                               embedding_size=embedding_size,
                               margin=math.degrees(margin),
                               scale=scale)

    pass




if __name__ == '__main__':
    compare()
