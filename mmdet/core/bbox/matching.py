import mmcv

from . import matchers


def build_matcher(cfg, **kwargs):
    if isinstance(cfg, matchers.BaseMatcher):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, matchers, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))
