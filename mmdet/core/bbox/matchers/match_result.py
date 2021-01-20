import torch


class MatchResult(object):

    def __init__(self, bboxes, bbox_inds, gt_labels):
        self.bboxes = bboxes
        self.bbox_inds = bbox_inds
        self.gt_labels = gt_labels
