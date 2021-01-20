import numpy as np
import torch.nn as nn
import torch

from mmcv.cnn import normal_init
from mmdet.core import force_fp32, bbox2delta, delta2bbox, bbox_overlaps, multi_apply, build_matcher, PseudoCollector

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from ..builder import build_loss
from .anchor_head import AnchorHead


@HEADS.register_module
class FreeAnchorHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 prior_prob=0.01,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_pos=dict(
                     type='FreeAnchorLoss', loss_weight=0.5),
                 loss_neg=dict(
                     type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.0, loss_weight=0.5),
                 **kwargs):
        self.prior_prob = prior_prob
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(FreeAnchorHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

        loss_pos.update(dict(get_bbox_prob_and_overlap=self.get_bbox_prob_and_overlap))
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_pos = build_loss(loss_pos)
        self.loss_neg = build_loss(loss_neg)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(self.prior_prob)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        gt_bboxes_ignore = (None,) * len(gt_bboxes) if gt_bboxes_ignore is None else gt_bboxes_ignore
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)

        def flatten(cls_score, bbox_pred):
            N, _, H, W = cls_score.shape

            cls_score_flattened = cls_score \
                .view(N, -1, self.cls_out_channels, H, W) \
                .permute(0, 3, 4, 1, 2) \
                .reshape(N, -1, self.cls_out_channels)

            bbox_pred_flattened = bbox_pred \
                .view(N, -1, 4, H, W) \
                .permute(0, 3, 4, 1, 2) \
                .reshape(N, -1, 4)

            return cls_score_flattened, bbox_pred_flattened

        cls_scores_flattened, bbox_preds_flattened = multi_apply(flatten, cls_scores, bbox_preds)
        cls_scores_collected = torch.cat(cls_scores_flattened, dim=1)
        bbox_preds_collected = torch.cat(bbox_preds_flattened, dim=1)

        anchors = tuple(map(torch.cat, anchor_list))
        cls_scores = cls_scores_collected.unbind(dim=0)
        bbox_preds = bbox_preds_collected.unbind(dim=0)

        bbox_matcher = build_matcher(cfg.matcher)
        bbox_collector = PseudoCollector(self.cls_out_channels)

        def match_collecting(anchors,
                             gt_bboxes,
                             gt_labels,
                             gt_bboxes_ignore,
                             cls_scores):
            match_result = bbox_matcher.match(anchors, gt_bboxes, gt_labels, gt_bboxes_ignore)
            collecting_result = bbox_collector.collect(match_result)

            return (
                collecting_result.bbox_inds,
                collecting_result.sparse_indices,
                collecting_result.neg_scores_mask
            )

        matched_inds, sparse_indices, neg_scores_mask = multi_apply(
            match_collecting,
            anchors,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            cls_scores
        )
        loss_pos, pos_inds = self.loss_pos(cls_scores, bbox_preds, gt_bboxes, gt_labels, anchors, matched_inds, sparse_indices)
        
        neg_scores = torch.cat([
            cls_score[torch.sparse_coo_tensor(pos_ind, torch.ones_like(pos_ind[0]), size=cls_score.size()).to_dense() == 0] 
            for cls_score, pos_ind in zip(cls_scores, pos_inds)
        ])[:, None]
        num_positives = cls_scores_collected.numel() - neg_scores.numel()
        
        def negative_bag_loss(logits, *args, **kwargs):
            weight = logits / (1 - logits).pow(2).clamp(min=1e-12)
            bag_prob = (weight * logits).sum(*args, **kwargs) / weight.sum(*args, **kwargs)

            return torch.nn.functional.binary_cross_entropy(
                bag_prob,
                torch.zeros_like(bag_prob),
                reduction='none'
            )

#         loss_neg = (
#             self.loss_neg.loss_weight * 
#             negative_bag_loss(torch.sigmoid(neg_scores)) / 
#             torch.softmax(neg_scores, dim=-1).min(dim=-1).values.detach()
#         )
#         if num_positives > 5000:
#             print('cls_scores_collected: {}, neg_scores: {}, gt_bboxes: {}, gt_labels: {}, pos_inds: {} {}'.format(
#                 cls_scores_collected.shape,
#                 neg_scores.shape,
#                 gt_bboxes.shape,
#                 gt_labels.shape,
#                 pos_inds[0].shape,
#                 pos_inds[1].shape,
#             ))
        loss_neg = self.loss_neg(neg_scores, torch.zeros_like(neg_scores, dtype=torch.long), avg_factor=num_positives)

        return dict(loss_pos=loss_pos, loss_neg=loss_neg)

    def get_bbox_prob_and_overlap(self, anchors, bbox_preds, gt_bboxes):
        bbox_targets = bbox2delta(
            anchors,
            gt_bboxes[:, None, :].expand_as(anchors),
            self.target_means,
            self.target_stds
        )
        bbox_prob = self.loss_bbox(bbox_preds, bbox_targets, reduction_override='none').sum(dim=-1).neg().exp()

        pred_boxes = delta2bbox(
            anchors,
            bbox_preds,
            self.target_means,
            self.target_stds
        )
        bbox_overlap = bbox_overlaps(gt_bboxes[:, None, :].expand_as(pred_boxes), pred_boxes, is_aligned=True)

        return bbox_prob, bbox_overlap