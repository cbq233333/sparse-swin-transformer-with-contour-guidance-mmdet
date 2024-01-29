# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage_edge import SingleStageDetectorEdge


@DETECTORS.register_module()
class FCOSEdge(SingleStageDetectorEdge):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FCOSEdge, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
