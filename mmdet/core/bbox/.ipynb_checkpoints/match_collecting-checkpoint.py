import mmcv

from . import matchers
from . import collectors


def build_matcher(cfg, **kwargs):
    if isinstance(cfg, matchers.BaseMatcher):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, matchers, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))


def build_collector(cfg, **kwargs):
    if isinstance(cfg, collectors.BaseCollector):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, collectors, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))


def match_and_collect(bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore, cfg):
    bbox_matcher = build_matcher(cfg.assigner)
    bbox_collector = build_collector(cfg.sampler)
    match_result = bbox_matcher.match(bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore)
    collecting_result = bbox_collector.collect(match_result, bboxes)
    return match_result, collecting_result
