import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import PointGenerator, bbox2distance, distance2bbox, bbox_overlaps, force_fp32, multi_apply, \
    multiclass_nms, build_matcher, PseudoCollector
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob


@HEADS.register_module
class FreePointHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 bboxes_per_octave=9,
                 strides=(8, 16, 32, 64, 128),
                 distance_norm=True,
                 loss_bbox=dict(
                     type='IoULoss', loss_weight=1.0),
                 loss_pos=dict(
                     type='FreeAnchorLoss', bbox_thr=0.0, loss_weight=0.5),
                 loss_neg=dict(
                     type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.0, loss_weight=0.5),
                 **kwargs):
        super(FreePointHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.bboxes_per_octave = bboxes_per_octave
        self.strides = strides
        self.distance_norm = distance_norm
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.point_generator = PointGenerator()

        loss_pos.update(dict(get_bbox_prob_and_overlap=self.get_bbox_prob_and_overlap))
        self.loss_bbox = build_loss(loss_bbox) if loss_bbox is not None else None
        self.loss_pos = build_loss(loss_pos)
        self.loss_neg = build_loss(loss_neg)

        self._init_layers()

    def _init_layers(self):
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
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.bboxes_per_octave * self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, self.bboxes_per_octave * 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = nn.functional.relu(scale(self.fcos_reg(reg_feat)).float())
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
        assert len(cls_scores) == len(bbox_preds)
        gt_bboxes_ignore = (None,) * len(gt_bboxes) if gt_bboxes_ignore is None else gt_bboxes_ignore
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        point_list = self.get_points(featmap_sizes, img_metas)

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

        points = tuple(map(torch.cat, point_list))
        cls_scores = cls_scores_collected.unbind(dim=0)
        bbox_preds = bbox_preds_collected.unbind(dim=0)

        bbox_matcher = build_matcher(cfg.matcher)
        bbox_collector = PseudoCollector(self.cls_out_channels)

        def match_collecting(points,
                             gt_bboxes,
                             gt_labels,
                             gt_bboxes_ignore,
                             cls_scores):
            match_result = bbox_matcher.match(points, gt_bboxes, gt_labels, gt_bboxes_ignore)
            collecting_result = bbox_collector.collect(match_result)

            return (
                collecting_result.bbox_inds,
                collecting_result.sparse_indices,
                cls_scores[collecting_result.neg_scores_mask]
            )

        matched_inds, sparse_indices, neg_scores = multi_apply(
            match_collecting,
            points,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            cls_scores
        )

        ##################################################
#         from matplotlib.colors import Normalize
#         import matplotlib.pyplot as plt
#         import matplotlib.image as image

#         with torch.no_grad():
#             show_filename = img_metas[0]['filename']
#             show_scale_factor = img_metas[0]['scale_factor']
#             show_flip = img_metas[0]['flip']
#             show_matched_ind = matched_inds[0]
#             show_points = points[0]
#             show_matched_points = show_points[show_matched_ind[0], :]
#             show_gt_bboxes = gt_bboxes[0]
#             show_matched_bbox_preds = bbox_preds[0][show_matched_ind[0], :]

#             show_bbox_cls_prob = cls_scores[0][show_matched_ind[0], gt_labels[0][0] - 1].sigmoid()
#             show_bbox_loc_prob = self.get_bbox_prob_and_overlap(show_matched_points[None, ...], show_matched_bbox_preds[None, ...], show_gt_bboxes[None, 0, :])[0]
#             show_bbox_prob = show_bbox_cls_prob * show_bbox_loc_prob

#             # show_bbox_prob /= show_bbox_prob.max()

#             show_image = image.imread(show_filename)[:, ::-1, :] if show_flip else image.imread(show_filename)
#             show_bbox = show_gt_bboxes[0].cpu().numpy() / show_scale_factor

#             plt.figure(figsize=(48, 36))
#             for i, s in enumerate(self.strides):
#                 ax = plt.subplot(1, len(self.strides), i + 1)
#                 plt.imshow(show_image)

#                 show_level_points = show_matched_points[(show_matched_points[:, 2] - s).abs() < 1, :2].cpu().numpy()
#                 show_level_prob = show_bbox_prob[0][(show_matched_points[:, 2] - s).abs() < 1].cpu().numpy()
#                 ax.add_patch(plt.Rectangle(
#                     (show_bbox[0], show_bbox[1]), show_bbox[2] - show_bbox[0], show_bbox[3] - show_bbox[1],
#                     fill=False, edgecolor='g', linewidth=2, alpha=0.5
#                 ))
#                 ax.scatter(
#                     show_level_points[:, 0] / show_scale_factor,
#                     show_level_points[:, 1] / show_scale_factor,
#                     s=s * 0.5, marker='o', c=show_level_prob,
#                     norm=Normalize(vmin=0., vmax=1.)
#                 )
#             plt.subplot(1, len(self.strides), 3)
#             plt.title('[{:.2f}, - {:.2f} -, {:.2f}]'.format(show_bbox_prob.min().item(), show_bbox_prob.mean().item(), show_bbox_prob.max().item()))
#             plt.show()
        ##################################################

        neg_scores = torch.cat(neg_scores, dim=0)[:, None]
        num_positives = cls_scores_collected.numel() - neg_scores.numel()

        loss_pos = self.loss_pos(cls_scores, bbox_preds, gt_bboxes, gt_labels, points, matched_inds, sparse_indices)
        loss_neg = self.loss_neg(neg_scores, torch.zeros_like(neg_scores, dtype=torch.long), avg_factor=num_positives)

        return dict(loss_pos=loss_pos, loss_neg=loss_neg)

    def get_points(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generator.grid_points(
                featmap_sizes[i], self.strides[i])
            points[..., :2] += self.strides[i] / 2
            multi_level_points.append(torch.repeat_interleave(points, self.bboxes_per_octave, dim=0))
        points_list = [multi_level_points for _ in range(num_imgs)]

        return points_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, img_metas)[0]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_points, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, points in zip(cls_scores, bbox_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            scores = cls_score.sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = distance2bbox(points, bbox_pred, norm=self.distance_norm, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels

    def get_bbox_prob_and_overlap(self, points, bbox_preds, gt_bboxes):

        bbox_targets = bbox2distance(
            points,
            gt_bboxes[:, None, :].repeat(1, points.shape[1], 1),
            norm=self.distance_norm
        )
        bbox_prob = self.loss_bbox(bbox_preds, bbox_targets, reduction_override='none').neg().exp()

        pred_boxes = distance2bbox(
            points,
            bbox_preds,
            norm=self.distance_norm
        )
        bbox_overlap = bbox_overlaps(gt_bboxes[:, None, :].expand_as(pred_boxes), pred_boxes, is_aligned=True)
#         bbox_overlap = bbox_prob

        return bbox_prob, bbox_overlap
