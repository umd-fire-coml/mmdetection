pimport torch.nn as nn

from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class ConvFCBBoxAlpDimHead(ConvFCBBoxHead):
    """Specific type of bbox head, with shared conv and fc layers and four optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs  /-> alp convs -> alp fcs -> alp
                                \-> dim convs -> dim fcs -> dim
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 with_alp=True, # new params added from here
                 with_dim=True,
                 num_alp_convs=0,
                 num_alp_fcs=0,
                 num_dim_convs=0,
                 num_dim_fcs=0,
                 loss_alp=dict(
                     type='AngularL2Loss',
                     beta=1.0,
                     loss_weight=1.0),
                 loss_dim=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     loss_weight=1.0),
                 *args,
                 **kwargs):
        super(ConvFCBBoxAlphaDimesionsHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + 
                num_cls_convs + num_cls_fcs + 
                num_reg_convs + num_reg_fcs + 
                num_alp_convs + num_alp_fcs +
                num_dim_convs + num_dim_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0 or num_alp_convs > 0 or num_dim_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        if not self.with_alp:
            assert num_alp_convs == 0 and num_alp_fcs == 0
        if not self.with_dim:
            assert num_dim_convs == 0 and num_dim_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # new params
        self.with_alp = with_alp
        self.with_dim = with_dim
        self.num_alp_convs = num_alp_convs
        self.num_alp_fcs = num_alp_fcs
        self.num_dim_convs = num_dim_convs
        self.num_dim_fcs = num_dim_fcs
        
        self.loss_alp = build_loss(loss_alp)
        self.loss_dim = build_loss(loss_dim)

        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)
        
        # add alp specific branch
        self.alp_convs, self.alp_fcs, self.alp_last_dim = \
            self._add_conv_fc_branch(
                self.num_alp_convs, self.num_alp_fcs, self.shared_out_channels)
        
        # add dim specific branch
        self.dim_convs, self.dim_fcs, self.dim_last_dim = \
            self._add_conv_fc_branch(
                self.num_dim_convs, self.num_dim_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_alp_fcs == 0:
                self.reg_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_dim_fcs == 0:
                self.reg_last_dim *= (self.roi_feat_size * self.roi_feat_size)

        self.relu = nn.ReLU(inplace=True)
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)
        # added the fcs for alp and dim
        if self.with_alp:
            out_dim_alp = (1 if self.reg_class_agnostic else 1 *
                           self.num_classes)
            self.fc_alp = nn.Linear(self.alp_last_dim, out_dim_alp)
        if self.with_dim:
            out_dim_dim = (3 if self.reg_class_agnostic else 3 *
                           self.num_classes)
            self.fc_dim = nn.Linear(self.dim_last_dim, out_dim_dim)

    def init_weights(self):
        super(ConvFCBBoxAlphaDimesionsHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs, self.alp_fcs, self.dim_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    # probably not working
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduce=True,
             alp_angle,
             dim_meter,
             alp_targets,
             alp_weights,
             dim_targets,
             dim_weights):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=avg_factor)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0))
        # added new loss for alp
        if alp_angle is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_alp_pred = alp_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_alp'] = ???
        # added new loss for dim
        if dim_meter is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_alp_pred = alp_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_dim'] = ???
        return losses
    
    # refining forward with alp 
    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
                
        # separate branches
        x_cls = x
        x_reg = x
        x_alp = x
        x_dim = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        
        # new branch added for alp
        for conv in self.alp_convs:
            x_alp = conv(x_alp)
        if x_alp.dim() > 2:
            if self.with_avg_pool:
                x_alp = self.avg_pool(x_alp)
            x_alp = x_alp.view(x_alp.size(0), -1)
        for fc in self.alp_fcs:
            x_alp = self.relu(fc(x_alp))
        
        # new branch added for dim
        for conv in self.dim_convs:
            x_dim = conv(x_dim)
        if x_dim.dim() > 2:
            if self.with_avg_pool:
                x_dim = self.avg_pool(x_dim)
            x_dim = x_dim.view(x_dim.size(0), -1)
        for fc in self.dim_fcs:
            x_dim = self.relu(fc(x_dim))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        
        # new output added for alp
        alp_angle = self.fc_alp(x_alp) if self.with_alp else None
        dim_meter = self.fc_dim(x_dim) if self.with_dim else None
        
        return cls_score, bbox_pred, alp_angle, dim_meter


@HEADS.register_module
class SharedFCBBoxAlpDimHead(ConvFCBBoxAlpDimHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxAlpDimHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            with_alp=True,
            with_dim=True,
            num_alp_convs=0,
            num_alp_fcs=0,
            num_dim_convs=0,
            num_dim_fcs=0,
            *args,
            **kwargs)
