import torch

from .match_result import MatchResult
from .base_matcher import BaseMatcher


class PointMatcher(BaseMatcher):

    def __init__(self,
                 num_matches,
                 gt_bbox_thr=None,
                 ignore_iof_thr=-1):
        self.num_matches = num_matches
        self.gt_bbox_thr = gt_bbox_thr
        self.ignore_iof_thr = ignore_iof_thr

    def match(self, points, gt_bboxes, gt_labels, gt_bboxes_ignore=None):

        if points.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            raise NotImplementedError

        num_points = points.shape[0]
        num_gts = gt_bboxes.shape[0]
        gt_labels = gt_labels - 1

        xs = points[:, None, 0].expand(num_points, num_gts)
        ys = points[:, None, 1].expand(num_points, num_gts)
        ss = points[:, None, 2].expand(num_points, num_gts)

        distance_xs = (2 * xs - gt_bboxes[:, 0] - gt_bboxes[:, 2]).abs() / (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1)
        distance_ys = (2 * ys - gt_bboxes[:, 1] - gt_bboxes[:, 3]).abs() / (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)

        distances = torch.max(distance_xs, distance_ys) / ss
        if self.gt_bbox_thr is not None:
            out_of_gts = (distance_xs > self.gt_bbox_thr) + (distance_ys > self.gt_bbox_thr)
            not_lowest = (ss > ss.min())
            distances[out_of_gts * not_lowest] = torch.finfo(distances.dtype).max

        _, bbox_inds = torch.topk(
            distances.t(), min(self.num_matches, num_points),
            dim=1, largest=False, sorted=False
        )
        match_result = MatchResult(points, bbox_inds, gt_labels)

        return match_result
