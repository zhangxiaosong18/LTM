import torch

from ..geometry import bbox_overlaps
from .match_result import MatchResult
from .base_matcher import BaseMatcher


class TopkIoUMatcher(BaseMatcher):

    def __init__(self,
                 num_matches,
                 ignore_iof_thr=-1,
                 add_gt_probability=-1):
        self.num_matches = num_matches
        self.ignore_iof_thr = ignore_iof_thr
        self.add_gt_probability = add_gt_probability

    def match(self, bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore=None):

        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            raise NotImplementedError

        if self.add_gt_probability >= 0:
            rand_sampling = torch.rand(gt_bboxes.shape[0], device=gt_bboxes.device) <= self.add_gt_probability
            bboxes = torch.cat([bboxes[:, :4], gt_bboxes[rand_sampling]], dim=0)
        else:
            bboxes = bboxes[:, :4]
        gt_labels = gt_labels - 1

        overlaps = bbox_overlaps(gt_bboxes, bboxes)
        _, bbox_inds = torch.topk(
            overlaps, min(self.num_matches, len(bboxes)),
            dim=1, sorted=False
        )
        match_result = MatchResult(bboxes, bbox_inds, gt_labels)

        return match_result
