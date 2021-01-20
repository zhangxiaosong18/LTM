from abc import ABCMeta, abstractmethod


class BaseMatcher(metaclass=ABCMeta):

    @abstractmethod
    def match(self, bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        pass
