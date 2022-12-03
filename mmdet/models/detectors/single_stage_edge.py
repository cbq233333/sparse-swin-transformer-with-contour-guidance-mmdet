# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import numpy
import cv2


@DETECTORS.register_module()
class SingleStageDetectorEdge(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetectorEdge, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetectorEdge, self).forward_train(img, img_metas)
        img_edge = []
        _,_,h,w = img.shape
        for i in range(len(img_metas)):
            img_path = img_metas[i]['filename']
            gt_bboxes_np = gt_bboxes[i]
            gt_bboxes_np = gt_bboxes_np.cpu().numpy()
            img_n = cv2.imread(img_path)
            h_ori,w_ori,_=img_n.shape
            img_np=numpy.pad(img_n, ((0,h-h_ori),(0,w-w_ori),(0,0)), 'constant')
            img_out = numpy.zeros((h,w),dtype=numpy.float32)
            #img_gray = cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY)
            #ret, img_binary = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
            #contours, hierarchy = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            for gt_bbox_np in gt_bboxes_np:
                x1 = int(gt_bbox_np[0])
                y1 = int(gt_bbox_np[1])
                x2 = int(gt_bbox_np[2])
                y2 = int(gt_bbox_np[3])
                #contours, hierarchy = cv2.findContours(img_binary[y1:y2,x1:x2],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                
                img_slice = img_np[y1:y2,x1:x2,:]
                #img_out[y1:y2,x1:x2,:] = 0
                blurred = cv2.GaussianBlur(img_slice,(11,11),0)
                edge = cv2.Canny(blurred, 10, 70)
                img_out[y1:y2,x1:x2]+=edge
                img_out=img_out/255
                # img_out=numpy.expand_dims(img_out,0)
                # img_out = torch.from_numpy(img_out)
                # img_out = img_out.to(img.device)
            #cv2.imwrite("/workspace/mmdetection/img_out.jpg",img_out)
            #cv2.imwrite("/home/cbq/mmdetection/img.jpg",img_np)
            #cv2.imwrite("/home/cbq/mmdetection/img_edge_ssdd.jpg",img_out*255)
            img_out_t = torch.from_numpy(img_out)
            img_edge.append(img_out_t)
        #img_edge = torch.from_numpy(img_edge)
        #img_edge.to(img.device)
        img_edge = torch.stack(img_edge,0)
        #img_edge=numpy.expand_dims(img_edge,1)
        #img_edge = torch.from_numpy(edge)
        img_edge = img_edge.to(img.device)
        img_edge = img_edge[:,None,:,:]


        # #blurred = cv2.GaussianBlur(img_np[y1:y2,x1:x2,:],(11,11),0)
        # #gaussImg = cv2.Canny(blurred, 10, 70)
        # img_e = img
        # img_e[y1:y2,x1:x2,:] = 0
        # img_e[y1:y2,x1:x2,:] += gaussImg[:,:,None]
        # cv2.imwrite("/workspace/mmdetection/edge.jpg",img_e)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_edge, img_metas, gt_bboxes,
                                             gt_labels, gt_masks, gt_bboxes_ignore)
        #losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                      gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
