# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import cv2
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, reduce_mean
from ..builder import HEADS, build_loss
from .anchor_free_head_edge import AnchorFreeHeadEdge

INF = 1e8




up_kwargs = {'mode': 'bilinear', 'align_corners': False}

@HEADS.register_module()


class FCOSHeadEdge(AnchorFreeHeadEdge):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_edge=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bitimg=dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_edge = build_loss(loss_edge)
        self.loss_bitimg = build_loss(loss_bitimg)
        #self.edge_head = EdgeHead(in_channels, in_index=[0])

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.sig_edge = nn.Sigmoid()
        self.conv_bitimg1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 1, 1, 0),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(True),
            nn.Conv2d(self.in_channels, self.in_channels, 1, 1, 0),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(True)
        )
        self.conv_bitimg2 = nn.Conv2d(self.in_channels, 1, 1, 1, 0)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats, img_edge=None):
        """Forward features from the upstream network.
        #edge写到这里?
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        #todo:在这里写预测二值图的网络，只输出一个！
        if img_edge is not None:
            size = []
            #_,_,h,w = feats[0].shape
            _,_,h,w = img_edge.shape
            #h *= 4
            #w *= 4
            size.append(h)
            size.append(w)
            x=feats[0]
            #if h==200:
            #size[0] *= stride
            #size[1] *= stride

            # feature = x[0,0, :, :]
            # for i in range(0,x.size()[1]):
            #     if i>0:
            #         continue
            #     else:
            #         feature+=x[0,i,:,:]
            # feature = feature[:,:,None]
            # feature = feature.squeeze()
            # feature = feature.numpy()
            # feature = cv2.cvtColor(feature, cv2.COLOR_RGB2BGR)
            # cv2.imwrite('/workspace/mmdetection/feature_map.jpg',feature)
            #可视化
            x_edge = F.interpolate(x, size, **up_kwargs)
            x_edge_f = x_edge + img_edge
            #x_edge_f = self.sig_edge(x_edge_f)
            x_edge_f_np=x_edge_f.detach().cpu().numpy()
            x_edge_f_out=[]
            for x_edge_fnp in x_edge_f_np:
                x_edge_fnp = x_edge_fnp.transpose(1,2,0)
                x_edge_fnp_single = x_edge_fnp[:,:,0]
                xc = cv2.Sobel(x_edge_fnp_single,-1,1,0)
                yc = cv2.Sobel(x_edge_fnp_single,-1,0,1)
                #absX = cv2.convertScaleAbs(xc) # 转回uint8
                #absY = cv2.convertScaleAbs(yc)
                dst = cv2.addWeighted(xc,0.5,yc,0.5,0)
                #dst = dst.transpose(2,0,1)
                dst = numpy.expand_dims(dst,0)
                dst = torch.from_numpy(dst)
                x_edge_f_out.append(dst)
            #x_edge_f_out= numpy.expand_dims(x_edge_f_out,1)
            #x_edge_f_out=torch.from_numpy(x_edge_f_out)
            x_edge_f_outs = torch.stack(x_edge_f_out,0)
            x_edge_f_outs = x_edge_f_outs.to(img_edge.device)#输出
            x_edge_f_outs = self.sig_edge(x_edge_f_outs)
            bit_input = x_edge*x_edge_f_outs+x_edge
            bit1 = self.conv_bitimg1(bit_input)
        #cv2.imwrite('/workspace/mmdetection/feature_edge.jpg',edge1[0])
            #bit1 = F.interpolate(bit1, size, **up_kwargs)
            bitimg = self.conv_bitimg2(bit1)
            #cv2.imwrite('/workspace/mmdetection/edge_pred.jpg',edge[0])
                #outputs.append(edge)
            
            cbc_outs = multi_apply(self.forward_single, feats, self.scales,
                            self.strides)
            cls_outs = cbc_outs[0]
            bbox_outs = cbc_outs[1]
            centerness_outs = cbc_outs[2]
        else:
            cbc_outs = multi_apply(self.forward_single, feats, self.scales,
                            self.strides)
            cls_outs = cbc_outs[0]
            bbox_outs = cbc_outs[1]
            centerness_outs = cbc_outs[2]
            x_edge_f_outs = None
            bitimg = None

        return cls_outs, bbox_outs, centerness_outs, x_edge_f_outs, bitimg

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        #edge module
        #img_path=img_mates['']
        #img=cv2.imread(img_path)
        #x_edge=img+x
        #blurred = cv2.GaussianBlur(img,(11,11),0)
        #gaussImg = cv2.Canny(blurred, 10, 70)

        # size = []
        # _,_,h,w = x.shape
        # h *= stride
        # w *= stride
        # size.append(h)
        # size.append(w)
        # #if h==200:
        # #size[0] *= stride
        # #size[1] *= stride

        # # feature = x[0,0, :, :]
        # # for i in range(0,x.size()[1]):
        # #     if i>0:
        # #         continue
        # #     else:
        # #         feature+=x[0,i,:,:]
        # # feature = feature[:,:,None]
        # # feature = feature.squeeze()
        # # feature = feature.numpy()
        # # feature = cv2.cvtColor(feature, cv2.COLOR_RGB2BGR)
        # # cv2.imwrite('/workspace/mmdetection/feature_map.jpg',feature)
        # if h==200:
        #     x_edge = F.interpolate(x, size, **up_kwargs)
        #     x_edge_f = x_edge + img_edge
        #     x_edge_f_np=x_edge_f.cpu().numpy()
        #     x_edge_f_out=[]
        #     for x_edge_fnp in x_edge_f_np:
        #         xc = cv2.Sobel(x_edge_fnp,cv2.CV_16S,1,0)
        #         yc = cv2.Sobel(x_edge_fnp,cv2.CV_16S,0,1)
        #         absX = cv2.convertScaleAbs(xc) # 转回uint8
        #         absY = cv2.convertScaleAbs(yc)
        #         dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
        #         x_edge_f_out.append(dst)
        #     x_edge_f_out= numpy.expand_dims(x_edge_f_out,1)
        #     x_edge_f_out=torch.from_numpy(x_edge_f_out)
        #     x_edge_f_out.to(x.device)#输出
        #     x_edge_f_outs=self.sig_edge(x_edge_f_out)
        #     bit_input = x_edge*x_edge_f_outs+x_edge
        #     bit1 = self.conv_bitimg1(bit_input)
        # #cv2.imwrite('/workspace/mmdetection/feature_edge.jpg',edge1[0])
        #     #bit1 = F.interpolate(bit1, size, **up_kwargs)
        #     bitimg = self.conv_bitimg2(bit1)
        # #cv2.imwrite('/workspace/mmdetection/edge_pred.jpg',edge[0])
        #     #outputs.append(edge)
        # else:
        #     x_edge_f_out = None
        #     bitimg  = None

        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        #edge_pred = self.edge
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        
        return cls_score, bbox_pred, centerness


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             edge,
             bitimg,
             gt_bboxes,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)
        
        #二值图和边缘真值
        #gt,h,w=gt_masks[0].masks.shape
        b,c,h,w = bitimg.shape
        #bitimgs = numpy.zeros((1,h,w))
        #bitimgs = torch.Tensor(bitimgs)
        gt_bitimgs = []
        gt_edges = []
        for gt_mask in gt_masks:
            gt_bitimg = numpy.zeros((h,w),dtype=numpy.float32)
            for gt_bitmap in gt_mask.masks:
                #bitmap = bitmap*255
                #h_ori,w_ori = gt_bitmap.shape
                #gt_bitmap=numpy.pad(gt_bitmap, ((0,h-h_ori),(0,w-w_ori)), 'constant')
                gt_bitimg += gt_bitmap
                #gt_bitimg_t = torch.from_numpy(gt_bitimg)
                #gt_edge_t = torch.from_numpy(gt_edge)
                #gt_bitimg = gt_bitimg/255
            gt_edge = self.groundtruth_edge(gt_bitimg)
            gt_bitimg_t = torch.from_numpy(gt_bitimg)
            gt_edge_t = torch.from_numpy(gt_edge)
            gt_bitimgs.append(gt_bitimg_t[None,:,:])
            gt_edges.append(gt_edge_t[None,:,:])
        flatten_bitimgs = torch.stack(gt_bitimgs,0).to(cls_scores[0].device)#(batch,1,h800,w800)
        flatten_edges = torch.stack(gt_edges,0).to(cls_scores[0].device)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
        
        #edge_0 = edge[:,0,:,:]
        
        loss_edge = self.loss_edge(edge,flatten_edges)
        loss_bitimg = self.loss_bitimg(bitimg,flatten_bitimgs)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_edge=loss_edge,
            loss_bitimg=loss_bitimg)

    def groundtruth_edge(self, label, edge_width=1):
        if len(label.shape) == 2:
            label = label[numpy.newaxis, ...]
        label = label.astype(numpy.int)
        b, h, w = label.shape
        edge = numpy.zeros(label.shape)

        # right
        edge_right = edge[:, 1:h, :]
        edge_right[(label[:, 1:h, :] != label[:, :h - 1, :])] = 1

        # up
        edge_up = edge[:, :, :w - 1]
        edge_up[(label[:, :, :w - 1] != label[:, :, 1:w])] = 1

        # upright
        edge_upright = edge[:, :h - 1, :w - 1]
        edge_upright[(label[:, :h - 1, :w - 1] != label[:, 1:h, 1:w])] = 1

        # bottomright
        edge_bottomright = edge[:, :h - 1, 1:w]
        edge_bottomright[(label[:, :h - 1, 1:w] != label[:, 1:h, :w - 1])] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
        for i in range(edge.shape[0]):
            edge[i] = cv2.dilate(edge[i], kernel)
        edge = edge.squeeze(axis=0)
        return edge

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """
        warnings.warn(
            '`_get_points_single` in `FCOSHead` will be '
            'deprecated soon, we support a multi level point generator now'
            'you can get points of a single level feature map '
            'with `self.prior_generator.single_level_grid_priors` ')

        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points
    


