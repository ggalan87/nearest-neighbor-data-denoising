from mmcls.models.builder import HEADS
from mmcls.models.heads.cls_head import ClsHead, BaseHead
from mmcls.models.heads.linear_head import LinearClsHead
from torch.nn import functional as F
from torch import nn
from losses import TripletLoss
from mmcls.models.builder import build_loss

@HEADS.register_module()
class IdentHead(BaseHead):

    def __init__(self, loss=dict(type='TripletLoss', loss_weight=1.0)):
        super().__init__()
        self.compute_loss = build_loss(loss)

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        # compute accuracy
        #acc = self.compute_accuracy(cls_score, gt_label)
        #assert len(acc) == len(self.topk)
        losses['loss'] = loss
        return losses

    def forward_train(self, x, gt_label):
        feat = F.normalize(x, p=2, dim=1)
        losses = self.loss(feat, gt_label)
        return losses

    def simple_test(self, x):
        # feat = torch.cat(tuple(x), 1)
        feat = F.normalize(x, p=2, dim=1)
        return feat


@HEADS.register_module()
class MultiHead(BaseHead):
    def __init__(self, losses={'cls': dict(type='CrossEntropyLoss', loss_weight=1.0), 'ident': dict(type='TripletLoss', loss_weight=1.0)}, topk=(1, )):
        super().__init__()
        self.cls_head = ClsHead(losses['cls'], topk)
        self.ident_head = IdentHead(losses['ident'])

    def forward_train(self, x, gt_label):
        cls_out = self.cls_head.forward_train(x, gt_label)
        ident_out = self.ident_head.forward_train(x, gt_label)
        cls_out['loss'] += ident_out['loss']
        return cls_out

    def simple_test(self, x):
        return self.cls_head.simple_test(x), self.ident_head.simple_test(x)


@HEADS.register_module()
class AngularPenaltyHead(BaseHead):
    pass