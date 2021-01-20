import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..registry import LOSSES
from mmdet.core import multi_apply
from .utils import weight_reduce_loss


class Clip(Function):
    @staticmethod
    def forward(ctx, x, a, b):
        return x.clamp(a, b)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return grad_output, None, None


clip = Clip.apply


@LOSSES.register_module
class MatchLoss(nn.Module):
    def __init__(self,
                 get_bbox_prob_and_overlap,
                 loss_weight=1.0):
        super(MatchLoss, self).__init__()
        self.get_bbox_prob_and_overlap = get_bbox_prob_and_overlap
        self.loss_weight = loss_weight

    def loss_single(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, anchors, matched_inds, sparse_indices):
        num_objects, bag_size = matched_inds.shape
        gt_labels = gt_labels - 1

        # shape: [i, j]
        matched_cls_prob = torch.gather(
            cls_scores[matched_inds], 2, gt_labels[:, None, None].repeat(1, bag_size, 1)
        ).squeeze(2).sigmoid()

        # shape: [i, j, 4]
        matched_anchors = anchors[matched_inds]
        matched_bbox_preds = bbox_preds[matched_inds]

        matched_box_prob, matched_box_overlap = self.get_bbox_prob_and_overlap(
            matched_anchors,
            matched_bbox_preds,
            gt_bboxes
        )
        positive = matched_box_overlap >= 0.5
        pos_inds = torch.stack([matched_inds[positive], gt_labels[:, None].expand_as(matched_inds)[positive]], dim=0)

        # shape: [i, j]
        object_box_prob = (
            matched_box_overlap / 
            matched_box_overlap.max(dim=1, keepdim=True).values.clamp(min=1e-12)
        ).clamp(min=0, max=1)

        # shape: [i, j, c]
        sparse_object_box_prob = torch.sparse_coo_tensor(
            sparse_indices,
            object_box_prob.view(-1),
            (num_objects, cls_scores.shape[0], cls_scores.shape[1]),
        )

        # shape: [j, c]
        max_box_prob = sparse_max_dim0(sparse_object_box_prob).to_dense()

        # shape: [i, j]
        matched_neg_prob = 1 - torch.gather(
            max_box_prob[matched_inds], 2, gt_labels[:, None, None].repeat(1, bag_size, 1)
        ).squeeze(2)

        # shape: [i]
        loss_rec = positive_bag_loss(matched_cls_prob * matched_box_prob, dim=1)
        loss_prc = negative_bag_loss(matched_cls_prob * matched_neg_prob, dim=1)

        return loss_rec, loss_prc, pos_inds

    def forward(self,
                cls_scores,
                bbox_preds,
                gt_bboxes,
                gt_labels,
                anchors,
                matched_inds,
                sparse_indices):
        loss_rec, loss_prc, pos_inds = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            gt_bboxes,
            gt_labels,
            anchors,
            matched_inds,
            sparse_indices
        )
        loss_rec, loss_prc = weight_reduce_loss(torch.cat(loss_rec)), weight_reduce_loss(torch.cat(loss_prc))

        return (loss_rec + loss_prc) * self.loss_weight, pos_inds


def positive_bag_loss(logits, *args, **kwargs):
    weight = 1 / clip(1 - logits, 1e-12, None)
    bag_prob = (weight * logits).sum(*args, **kwargs) / weight.sum(*args, **kwargs)

    return F.binary_cross_entropy(
        bag_prob,
        torch.ones_like(bag_prob),
        reduction='none'
    )


def negative_bag_loss(logits, *args, **kwargs):
    weight = logits / clip((1 - logits).pow(2), 1e-12, None)
    bag_prob = (weight * logits).sum(*args, **kwargs) / weight.sum(*args, **kwargs)

    return F.binary_cross_entropy(
        bag_prob,
        torch.zeros_like(bag_prob),
        reduction='none'
    )


def sparse_max_dim0(sparse_tensor):
    indices, values = sparse_tensor._indices(), sparse_tensor._values()
    new_indices, inverse = torch.unique(indices[1:], dim=1, sorted=False, return_inverse=True)
    unique_indices = torch.cat([indices[0:1], inverse[None, :]], dim=0)
    new_values = torch.sparse_coo_tensor(unique_indices, values).to_dense().max(dim=0).values

    return torch.sparse_coo_tensor(new_indices, new_values)
