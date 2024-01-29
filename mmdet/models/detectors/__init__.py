# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseDetector
from .single_stage_edge import SingleStageDetectorEdge
from .fcos_edge import FCOSEdge

__all__ = [
    'BaseDetector', 'SingleStageDetectorEdge','FCOSEdge'
]
