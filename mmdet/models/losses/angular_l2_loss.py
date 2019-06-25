import torch
import torch.nn as nn
import math

from .utils import weighted_loss
from ..registry import LOSSES


@weighted_loss
def angular_l2_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.acos(torch.cos(pred - target)) / math.pi # here 
    loss = diff / beta
    return loss


@LOSSES.register_module
class AngularL2Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(AngularL2Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        loss_anglar = self.loss_weight * angular_l2_loss(
            pred,
            target,
            beta=self.beta,
            reduction=self.reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
