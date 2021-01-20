from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    for dataset in datasets:
        for sample in dataset:
            filename = sample['img_meta'].data['filename']
            scale_factor = sample['img_meta'].data['scale_factor']
            flip = sample['img_meta'].data['flip']
            gt_bboxes = sample['gt_bboxes'].data
            model.bbox_head.point_generator()
            point_list = model.bbox_head.get_points(featmap_sizes, img_metas)
            pass


if __name__ == '__main__':
    main()
