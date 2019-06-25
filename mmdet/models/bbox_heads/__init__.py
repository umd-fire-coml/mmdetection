from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead, SharedFCBBoxHead
#from .convfc_bbox_alp_dim_head import ConvFCBBoxAlpDimHead, SharedFCBBoxAlpDimHead # added custom heads

__all__ = ['BBoxHead', 'ConvFCBBoxHead', 'SharedFCBBoxHead'] # added custom heads
