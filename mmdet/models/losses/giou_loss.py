import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def giou_loss(pred, target):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted boundaries (left, top, right, bottom),
            shape (..., 4).
        target (Tensor): Corresponding gt bboxes, shape (..., 4).
    Return:
        Tensor: Loss tensor.
    """
    pred_left = pred[..., 0]
    pred_top = pred[..., 1]
    pred_right = pred[..., 2]
    pred_bottom = pred[..., 3]

    target_left = target[..., 0]
    target_top = target[..., 1]
    target_right = target[..., 2]
    target_bottom = target[..., 3]

    target_area = (target_left + target_right + 1.0) * (target_top + target_bottom + 1.0)
    pred_area = (pred_left + pred_right + 1.0) * (pred_top + pred_bottom + 1.0)

    w_intersect = (torch.min(pred_left, target_left) + torch.min(pred_right, target_right)).clamp(min=0.0) + 1.0
    h_intersect = (torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)).clamp(min=0.0) + 1.0

    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect

    g_w_intersect = (torch.max(pred_left, target_left) + torch.max(pred_right, target_right)).clamp(min=0.0) + 1.0
    g_h_intersect = (torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)).clamp(min=0.0) + 1.0

    ac_uion = g_w_intersect * g_h_intersect

    ious = area_intersect / area_union
    gious = ious - (ac_uion - area_union) / ac_uion
    losses = 1 - gious
    return losses


@LOSSES.register_module
class GIoULoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
