# Copyright (c) OpenMMLab. All rights reserved.
from .anchor_free_head import AnchorFreeHead
from .anchor_head import AnchorHead
from .fcos_head import FCOSHead
from .fcos_head_edge import FCOSHeadEdge
from .fcos_edge_head_hrsid import FCOSHeadEdgeHR

__all__ = [
    'AnchorFreeHead', 'AnchorHead', 'FCOSHead', 'FCOSHeadEdge','FCOSHeadEdgeHR' 
]
