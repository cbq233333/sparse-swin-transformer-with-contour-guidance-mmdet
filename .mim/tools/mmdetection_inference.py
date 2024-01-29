# -*- coding:utf-8 -*-
import os
import argparse
import cv2
import mmcv
import numpy as np
from pycocotools.coco import COCO, maskUtils
from mmdet.apis import init_detector, inference_detector
from mmdet.core import encode_mask_results
from temp_coco_utils import coco_eval
 
'''
用于生成可视化时候不同类别的颜色
'''
import colorsys
import random
def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors
 
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors
 
'''
用于将结果保存到json文件
'''
def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]
 
def proposal2json(coco, results):
 
    imgIds = coco.getImgIds(catIds=[])
    categories = coco.dataset['categories']
 
    json_results = []
    for idx in range(len(imgIds)):
        img_id = imgIds[idx]
 
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results
 
def det2json(coco, results):
 
    imgIds = coco.getImgIds(catIds=[])
    categories = coco.dataset['categories']
 
    json_results = []
    for idx in range(len(imgIds)):
        img_id = imgIds[idx]
 
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = categories[label]['id']
                json_results.append(data)
    return json_results
 
def segm2json(coco, results):
 
    imgIds = coco.getImgIds(catIds=[])
    categories = coco.dataset['categories']
 
    bbox_json_results = []
    segm_json_results = []
    for idx in range(len(imgIds)):
        img_id = imgIds[idx]
 
        det, seg = results[idx]
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = categories[label]['id']
                bbox_json_results.append(data)
 
            # segm results
            # some detectors use different score for det and segm
            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = categories[label]['id']
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results
 
'''
路径创建函数
'''
def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)
 
def main(args):
    eval_types = args.eval
    mkdir_os(args.output_vis_result)
    #异常为input图像没有GT-object,是一张纯背景图
    mkdir_os(os.path.join(args.output_vis_result, "recall", "abnormalcase"))
    mkdir_os(os.path.join(args.output_vis_result, "recall", "case"))
    mkdir_os(os.path.join(args.output_vis_result, "recall", "badcase"))
 
    mkdir_os(os.path.join(args.output_vis_result, "precision", "abnormalcase"))
    mkdir_os(os.path.join(args.output_vis_result, "precision", "case"))
    mkdir_os(os.path.join(args.output_vis_result, "precision", "badcase"))
 
    score_thr = 0.3
    model = init_detector(args.input_config_file, args.input_checkpoint_file, device='cuda:0')
    model.eval()
 
    '''
    生成可视化类别颜色,colorbar-第一个颜色是backgroud,保留项不做使用
    '''
    cnum = 8
    self_color = ncolors(cnum)
    colorbar_vis = np.zeros((cnum * 30, 100, 3), dtype=np.uint8)
    for ind, colo in enumerate(self_color):
        k_tm = np.ones((30, 100, 3), dtype=np.uint8) * np.array([colo[-1], colo[-2], colo[-3]])
        colorbar_vis[ind * 30:(ind + 1) * 30, 0:100] = k_tm
    cv2.imwrite('../colorbar_vis.png', colorbar_vis)
 
    coco = COCO(args.input_test_json)
    imgIds = coco.getImgIds(catIds=[])
    categories = coco.dataset['categories']
 

    results = []
    vis_imgpath = []
    vis_imgid = []
    num = 0
    count = len(imgIds)
    for idx in range(len(imgIds)):
        result_list = []
        print(num,'/',count)
        num += 1
 
        img_id = imgIds[idx]
        img_info = coco.loadImgs(img_id)[0]
 
        file_name = img_info['file_name']
        img_path = os.path.join(args.input_test_img_path, file_name)
        result = inference_detector(model, img_path)
        # result_list.append(result)
        # if isinstance(result_list[0], tuple):
        #     result = [(bbox_results, encode_mask_results(mask_results))
        #               for bbox_results, mask_results in result_list]
        # elif isinstance(result_list[0], dict) and 'ins_results' in result_list[0]:
        #     for j in range(len(result)):
        #         bbox_results, mask_results = result_list[j]['ins_results']
        #         result_list[j]['ins_results'] = (bbox_results,
        #                                     encode_mask_results(mask_results))
        # results.extend(result)
        results.append(result)
        vis_imgpath.append(img_path)
        vis_imgid.append(img_info['id'])
 
    if eval_types:
        print('Starting evaluate {}'.format(' and '.join(eval_types)))
        if eval_types == ['proposal_fast']:
            result_file = os.path.join("./result", "result_out.pkl")
            # 2021.1.13 by ynh
            recall_result, precision_result, coco_recall_result, coco_precision_result = coco_eval(result_file, eval_types, coco)
        else:
            if not isinstance(results[0], dict):
                out_file = os.path.join("./result", "result_out.pkl")
                result_files = dict()
                #faster_rcnn_r50_fpn_1x.py 走该分支 eval_types=['bbox']
                if isinstance(results[0], list):
                    json_results = det2json(coco, results)
                    result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
                    result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
                    mmcv.dump(json_results, result_files['bbox'])
                # mask_rcnn_r50_fpn_1x.py 走该分支 eval_types=['bbox','segm']
                elif isinstance(results[0], tuple):
                    json_results = segm2json(coco, results)
                    result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
                    result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
                    result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
                    mmcv.dump(json_results[0], result_files['bbox'])
                    mmcv.dump(json_results[1], result_files['segm'])
                elif isinstance(results[0], np.ndarray):
                    json_results = proposal2json(coco, results)
                    result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
                    mmcv.dump(json_results, result_files['proposal'])
                # 2021.1.13 by ynh
                recall_result, precision_result, coco_recall_result, coco_precision_result = coco_eval(result_files, eval_types, coco)
            else:
                for name in results[0]:
                    out_file = os.path.join("./result", "result_out.pkl")
                    print('\nEvaluating {}'.format(name))
                    outputs_ = [out[name] for out in results]
                    out_file = out_file + '.{}'.format(name)
                    result_files = dict()
                    if isinstance(outputs_[0], list):
                        json_results = det2json(coco, outputs_)
                        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
                        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
                        mmcv.dump(json_results, result_files['bbox'])
                    elif isinstance(outputs_[0], tuple):
                        json_results = segm2json(coco, outputs_)
                        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
                        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
                        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
                        mmcv.dump(json_results[0], result_files['bbox'])
                        mmcv.dump(json_results[1], result_files['segm'])
                    elif isinstance(outputs_[0], np.ndarray):
                        json_results = proposal2json(coco, outputs_)
                        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
                        mmcv.dump(json_results, result_files['proposal'])
                    # 2021.1.13 by ynh
                    recall_result, precision_result, coco_recall_result, coco_precision_result = coco_eval(result_files, eval_types, coco)
 
 
 
 
 
    #只关注bbox的recall
    print("\n", "单个类别进行评估", "\n")
    coco_recall = coco_recall_result[0]
    iStr = ' Categories={:>12s} | {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Recall'
    typeStr = '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(0.50, 0.95)
    areaRng = 'all'
    maxDets = 100
    # dimension of recall: [TxKxAxM]
    s = coco_recall
    s = s[:]
    for m_ind, m_cls in enumerate(categories):
        temp_s = s[:, m_ind, 0, 2]
        if len(temp_s[temp_s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(temp_s[temp_s > -1])
        print(iStr.format(m_cls['name'], titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
 
    coco_precision = coco_precision_result[0]
    iStr = ' Categories={:>12s} | {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision'
    typeStr = '(AP)'
    iouStr = '{:0.2f}:{:0.2f}'.format(0.50, 0.95)
    areaRng = 'all'
    maxDets = 100
    # dimension of precision: [TxRxKxAxM]
    s = coco_precision
    # IoU
    s = s[:]
    for m_ind, m_cls in enumerate(categories):
        temp_s = s[:, :, m_ind, 0, 2]
        if len(temp_s[temp_s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(temp_s[temp_s > -1])
        print(iStr.format(m_cls['name'], titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
 
 
 
 
    print("\n","数据分类:","\n")
    #只关注bbox的recall
    type_recall = np.array(recall_result[0])
    # 2021.1.13 by ynh
    type_precision = np.array(precision_result[0])
    catId_num, imgId_num = type_recall.shape[:2]
    vis_idx = 0
    num = 0
    for n_key in range(imgId_num):
        print(num,'/',imgId_num)
        num += 1
        imgId_recall = type_recall[:,n_key]
        # 2021.1.13 by ynh
        imgId_precision = type_precision[:,n_key]
        #召回率
        if ((imgId_recall > 0).sum())==0:
            recall = -1
        else:
            recall = np.sum(imgId_recall[imgId_recall > 0]) / (imgId_recall > 0).sum()
 
        # 2021.1.13 by ynh
        #准确率
        if ((imgId_precision > 0).sum())==0:
            precision = -1
        else:
            precision = np.sum(imgId_precision[imgId_precision > 0]) / (imgId_precision > 0).sum()
 
 
        #可视化流程
        result = results[n_key]
 
        # 判断bbox和segm
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        #为bbox添加类别
        for m_key, m_val in enumerate(bbox_result):
            if m_val.shape[:2][0] > 0:
                rows, clos = m_val.shape[:2]
                m_temp = np.ones((rows, 1), dtype=np.float32)*m_key
                bbox_result[m_key] = np.hstack((m_val, m_temp))
            else:
                bbox_result[m_key] = np.empty(shape=(0, 6), dtype=np.float32)
        bboxes = np.vstack(bbox_result)
 
        if score_thr > 0:
            assert bboxes.shape[1] == 6
            #这里需要修改-----------
            #scores = bboxes[:, -1]
            scores = bboxes[:, -2]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
 
        # 用于通过左右方式显示原图和可视化图
        img = mmcv.imread(vis_imgpath[n_key])
        img = img.copy()
        oriimg = img.copy()
 
        # 画出mask
        mask_list = []
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            #这里需要修改-----------
            #inds = np.where(bboxes[:, -1] > score_thr)[0]
            inds = np.where(bboxes[:, -2] > score_thr)[0]
            np.random.seed(42)
            color_masks = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for i in inds:
                i = int(i)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img[mask] = img[mask] * 0.5 + color_masks * 0.5
                mask_list.append(mask)
 
        #画出bbox
        font_scale = 0.8
        thickness = 4
        #bbox_color = (0, 255, 0)
        #text_color = (0, 255, 0)
        for bbox in bboxes:
            #通过bbox[-1]获取颜色color_id = (id+1)
            bbox_color = self_color[int(bbox[-1]+1)][::-1]
            text_color = self_color[int(bbox[-1]+1)][::-1]
 
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, bbox_color, thickness=thickness)
            if len(bbox) > 4:
                label_text = '{:.02f}'.format(bbox[-1])
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 5),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
 
        # 显示GT
        annIds = coco.getAnnIds(imgIds=vis_imgid[n_key], catIds=[], iscrowd=None)
        anns = coco.loadAnns(annIds)
 
        polygons = []
        color = []
        category_id_list = []
        for ann in anns:
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                        poly_list = poly.tolist()
                        polygons.append(poly_list)
 
                        # rgb-bgr
                        # mylist[start:end:step]
                        # 切片逆序[::-1]
                        if ann['iscrowd'] == 0 and ann["ignore"] == 0:
                            temp = self_color[ann['category_id']]
                            color.append(temp[::-1])
                        if ann['iscrowd'] == 1 or ann["ignore"] == 1:
                            temp = self_color[-1]
                            color.append(temp[::-1])
                        category_id_list.append(ann['category_id'])
                else:
                    print("error type(ann['segmentation']) != list")
                    exit()
 
        point_size = 2
        thickness = 4
        for key in range(len(polygons)):
            ndata = polygons[key]
            cur_color = color[key]
            label_id = category_id_list[key]
 
            label = 'error'
            for m_id in categories:
                if m_id['id']==label_id:
                    label = m_id['name']
 
            #segmentation
            if len(ndata)>2:
                for k in range(len(ndata)):
                    data = ndata[k]
                    cv2.circle(oriimg, (int(data[0]), int(data[1])), point_size, (cur_color[0], cur_color[1], cur_color[2]),
                               thickness)
            else:#bbox
                cv2.rectangle(oriimg, (int(ndata[0][0]), int(ndata[0][1])), (int(ndata[1][0]), int(ndata[1][1])),
                              (cur_color[0], cur_color[1], cur_color[2]),
                              thickness)
                cv2.putText(oriimg, label, (int(ndata[0][0]), int(ndata[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (cur_color[0], cur_color[1], cur_color[2]), 2)
 
        # 可视化显示mask + bbox
        h1, w1 = oriimg.shape[:2]
        h2, w2 = img.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1, :] = oriimg
        vis[:h2, w1:w1 + w2, :] = img
 
        # 2021.1.13 by ynh
        vis_idx += 1
        if (0< recall < 1):
            out_file = os.path.join(args.output_vis_result, "recall", "badcase", 'recall_result_{}.jpg'.format(vis_idx))
        elif (recall == -1):
            out_file = os.path.join(args.output_vis_result, "recall", "abnormalcase", 'result_{}.jpg'.format(vis_idx))
        else:
            out_file = os.path.join(args.output_vis_result, "recall", "case", 'result_{}.jpg'.format(vis_idx))
        cv2.imwrite(out_file, vis)
 
        # 2021.1.13 by ynh
        if (0< precision < 1):
            out_file = os.path.join(args.output_vis_result, "precision", "badcase", 'precision_result_{}.jpg'.format(vis_idx))
        elif (precision == -1):
            out_file = os.path.join(args.output_vis_result, "precision", "abnormalcase", 'result_{}.jpg'.format(vis_idx))
        else:
            out_file = os.path.join(args.output_vis_result, "precision", "case", 'result_{}.jpg'.format(vis_idx))
        cv2.imwrite(out_file, vis)
 
 
if __name__ == "__main__":
    '''
    /mmdet/core/evaluation/recall.py
    /mmdet/core/evaluation/mean_ap.py
    /mmdet/core/evaluation/eval_hooks.py
    /mmdet/core/evaluation/coco_utils.py
    /mmdet/core/evaluation/class_names.py
    /mmdet/core/evaluation/bbox_overlaps.py
    
    /lib/python3.6/site-packages/pycocotools/cocoeval.py
    /lib/python3.6/site-packages/pycocotools/coco.py
    /lib/python3.6/site-packages/pycocotools/mask.py
    '''
    parser = argparse.ArgumentParser(
        description=
        "计算 mmdetection中的 mAP和AP")
    # parser.add_argument('-icf',
    #                     "--input_config_file",
    #                     default='./configs/ssd/ssd512_coco.py',
    #                     help="set input_config_file")
    # parser.add_argument('-icf',
    #                     "--input_config_file",
    #                     default='./configs/fcos/ab_fcos_noaug.py',
    #                     help="set input_config_file")
    # parser.add_argument('-icf',
    #                     "--input_config_file",
    #                     default='./configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_1x_coco.py',
    #                     help="set input_config_file")
    # parser.add_argument('-icf',
    #                     "--input_config_file",
    #                     default='./configs/centernet/centernet_resnet18_140e_coco.py',
    #                     help="set input_config_file")
    parser.add_argument('-icf',
                        "--input_config_file",
                        default='./configs/aa_fcos_new/deswin_conv_shift_fcos.py',
                        help="set input_config_file")
    # parser.add_argument('-icp',
    #                     "--input_checkpoint_file",
    #                     default='./work_com/centernet_SSDD/epoch_40.pth',
    #                     help="set input_checkpoint_file")
    parser.add_argument('-icp',
                        "--input_checkpoint_file",
                        default='./work_dirs/work_aa_fcos_new/deswin_conv_shift_fcos_HRSID/epoch_200.pth',
                        help="set input_checkpoint_file")
    # parser.add_argument('-icp',
    #                     "--input_checkpoint_file",
    #                     default='./work_dirs/ab_fcos_noaug_HRSID/epoch_200.pth',
    #                     help="set input_checkpoint_file")
    # parser.add_argument('-icp',
    #                     "--input_checkpoint_file",
    #                     default='./work_com/faster_rcnn_dc5_SSDD/epoch_200.pth',
    #                     help="set input_checkpoint_file")
    # parser.add_argument('-itj',
    #                     "--input_test_json",
    #                     #default='annotations/batch4-ZD-data_instances_test2017.json',
    #                     default='/media/zxf/20dddce9-0f1e-4408-a57a-f8a33aadd1b9/home/cbq/Dataset/HRSID/annotations/instances_val2017.json',
    #                     help="set input_test_json")
    parser.add_argument('-itj',
                        "--input_test_json",
                        # default='annotations/batch4-ZD-data_instances_test2017.json',
                        default='/media/zxf/20dddce9-0f1e-4408-a57a-f8a33aadd1b9/home/cbq/Dataset/HRSID/annotations/instances_val2017.json',
                        help="set input_test_json")
    # parser.add_argument('-itp',
    #                     "--input_test_img_path",
    #                     default='/media/zxf/20dddce9-0f1e-4408-a57a-f8a33aadd1b9/home/cbq/Dataset/HRSID/test2017',
    #                     help="set input_test_img_path")
    parser.add_argument('-itp',
                        "--input_test_img_path",
                        default='/media/zxf/20dddce9-0f1e-4408-a57a-f8a33aadd1b9/home/cbq/Dataset/HRSID/test2017',
                        help="set input_test_img_path")
    parser.add_argument('-ovt',
                        "--output_vis_result",
                        default='./result_new',
                        help="set output vis")
    parser.add_argument('--eval',
                        type=str,
                        nargs='+',
                        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
                        default=['bbox'],
                        help='eval types')
    args = parser.parse_args()
 
    if args.output_vis_result is None:
        parser.print_help()
        exit()
 
    main(args)