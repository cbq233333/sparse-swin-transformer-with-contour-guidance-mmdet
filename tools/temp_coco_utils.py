import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
 
from temp_recall import eval_recalls
 
 
def coco_eval(result_files, result_types, coco, max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
        ]
 
    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)
 
    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_files, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return
 
    recall_list = [[] for _ in range(len(result_types))]
    # add 2021.1.13 by ynh
    precision_list = [[] for _ in range(len(result_types))]
    coco_recall_list = [[] for _ in range(len(result_types))]
    coco_precision_list = [[] for _ in range(len(result_types))]
    
    for m_key, res_type in enumerate(result_types):
        result_file = result_files[res_type]
        assert result_file.endswith('.json')
 
        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        #对给定的图像运行逐图像计算并将结果(dict列表)存储在self.evalImgs
        cocoEval.evaluate()
        #累积每个图像的评估结果并将结果存储到self.eval
        cocoEval.accumulate()
        #计算并显示评估结果,仅应用于默认参数设置
        cocoEval.summarize()
 
        #https://www.aiuai.cn/aifarm854.html
        #cocoEval.eval, cocoEval.evalImgs
 
        '''
        dimension of precision: [TxRxKxAxM]
        参考:https://zhuanlan.zhihu.com/p/60707912
        cocoEval.eval['precision']是一个5维的数组
        precision  - [TxRxKxAxM] precision for every evaluation setting
        catIds     - [all] K cat ids to use for evaluation
        iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
        recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
        areaRng    - [...] A=4 object area ranges for evaluation
        maxDets    - [1 10 100] M=3 thresholds on max detections per image
        第一维T：IoU的10个阈值，从0.5到0.95间隔0.05
        第二维R：101个recall 阈值，从0到101
        第三维K：类别，如果是想展示第一类的结果就设为0
        第四维A：area 目标的大小范围 （all，small, medium, large）（全部，小，中，大）
        第五维M：maxDets 单张图像中最多检测框的数量 三种 1,10,100
        coco_eval.eval['precision'][0, :, 0, 0, 2] 所表示的就是当IoU=0.5时
        从0到100的101个recall对应的101个precision的值
        '''
 
        #evalImgs每张图片的检测质量
        #eval整个数据集上的聚合检测质量
        '''
        return {
            'image_id':     imgId,
            'category_id':  catId,
            'aRng':         aRng,
            'maxDet':       maxDet,
            'dtIds':        [d['id'] for d in dt],
            'gtIds':        [g['id'] for g in gt],
            'dtMatches':    dtm,
            'gtMatches':    gtm,
            'dtScores':     [d['score'] for d in dt],
            'gtIgnore':     gtIg,
            'dtIgnore':     dtIg,
        }
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
         for catId in catIds
         for areaRng in p.areaRng
         for imgId in p.imgIds
        ]
        
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        '''
 
        #265
 
        p = cocoEval.params
        catIds = p.catIds if p.useCats else [-1]
        recalls = [[-1 for _ in range(len(p.imgIds))] for _ in range(len(catIds))]
        # add 2021.1.13 by ynh
        precision = [[-1 for _ in range(len(p.imgIds))] for _ in range(len(catIds))]
        #类别
        for m_catId, catId in enumerate(catIds):
            #图片
            object_TP_sum = 0
            object_FP_sum = 0
            recall_TPFN_sum = 0
            precision_TPFP_sum = 0
            for n_imgId, imgId in enumerate(p.imgIds):
                
                if p.useCats:
                    gt = cocoEval._gts[imgId, catId]
                    dt = cocoEval._dts[imgId, catId]
                else:
                    gt = [_ for cId in p.catIds for _ in cocoEval._gts[imgId, cId]]
                    dt = [_ for cId in p.catIds for _ in cocoEval._dts[imgId, cId]]
                if len(gt) == 0 and len(dt) == 0:
                    continue
 
                for g in gt:
                    if g['ignore']:
                        g['_ignore'] = 1
                    else:
                        g['_ignore'] = 0
 
                # sort dt highest score first, sort gt ignore last
                gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
                gt = [gt[i] for i in gtind]
                dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
                #取最大的框数目100
                dt = [dt[i] for i in dtind[0:100]]
                iscrowd = [int(o['iscrowd']) for o in gt]
                # load computed ious
                ious = cocoEval.ious[imgId, catId][:, gtind] if len(cocoEval.ious[imgId, catId]) > 0 else cocoEval.ious[imgId, catId]
 
                T = 1 # 我们只采用IOU阈值为0.5
                G = len(gt)
                D = len(dt)
                gtm = np.zeros((T, G))
                dtm = np.zeros((T, D))
                gtIg = np.array([g['_ignore'] for g in gt])
                dtIg = np.zeros((T, D))
                if not len(ious) == 0:
                    recall_TPFN = len(gt)
                    # add 2021.1.13 by ynh
                    precision_TPFP = len(dt)
                    # revise 2021.1.13 by ynh
                    object_TP = 0
                    object_FP = 0
                    exsit_gt = []
                    for dind, d in enumerate(dt):
                        if d['score']<0.3:
                            continue
                        
                        iou_thr = 0.5
                        m   = -1
                        for gind, g in enumerate(gt):
                            # if this gt already matched, and not a crowd, continue
                            # 我们只采用IOU阈值为0.5
                            if gtm[0,gind]>0 and not iscrowd[gind]:
                                continue
                            # if dt matched to reg gt, and on ignore gt, stop
                            if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                                break
                            # continue to next gt unless better match made
                            if ious[dind,gind] < iou_thr:
                                continue
                            # if match successful and best so far, store appropriately
                            iou_thr=ious[dind,gind]
                            m=gind
                        # if match made store id of match for both dt and gt
                        if m ==-1:
                            #if d['score']<0.5:
                                #continue
                            #else:
                            object_FP += 1
                            continue
                        else:
                            # revise 2021.1.13 by ynh
                            #exsit_gt.append(m)
                            ex=0
                            if exsit_gt is not None:                                
                                for gi in exsit_gt:
                                    if gi == m:
                                        ex = 1
                                if ex == 0:
                                    exsit_gt.append(m)
                                    object_TP += 1
                                else:
                                    #if d['score']<0.5:
                                        #continue
                                    #else:
                                    object_FP += 1
                                    continue
                            else:
                                exsit_gt.append(m)            
                                object_TP += 1

                    object_TP_sum += object_TP
                    object_FP_sum += object_FP
                    recall_TPFN_sum += recall_TPFN
                    precision_TPFP_sum += precision_TPFP
                    # revise 2021.1.13 by ynh
                    recalls[m_catId][n_imgId] = float(object_TP/recall_TPFN)
                    # add 2021.1.13 by ynh
                    precision[m_catId][n_imgId] = float(object_TP/precision_TPFP)
                    #print(recalls[m_catId][n_imgId])
                    #print(precision[m_catId][n_imgId])
        Recall = float(object_TP_sum/recall_TPFN_sum)
    
        #Precision = float(object_TP_sum/precision_TPFP_sum)
        Precision2 = float(object_TP_sum/(object_TP_sum+object_FP_sum))
        F1 = Recall * Precision2 / (Recall + Precision2) * 2
        print('recall:%f'%Recall)
        #print(Precision)
        print('precision:%f'%Precision2)
        print('F1:%f'%F1)
        recall_list[m_key] = recalls
        # add 2021.1.13 by ynh
        precision_list[m_key] = precision
 
        # dimension of recall: [TxKxAxM]
        s1 = cocoEval.eval['recall']
        # dimension of precision: [TxRxKxAxM]
        s2 = cocoEval.eval['precision']
        coco_recall_list[m_key] = s1
        coco_precision_list[m_key] = s2
    
    

    return recall_list, precision_list, coco_recall_list, coco_precision_list
 
 
def fast_eval_recall(results,
                     coco,
                     max_dets,
                     iou_thrs=np.arange(0.5, 0.96, 0.05)):
    if mmcv.is_str(results):
        assert results.endswith('.pkl')
        results = mmcv.load(results)
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a filename, not {}'.
            format(type(results)))
 
    gt_bboxes = []
    img_ids = coco.getImgIds()
    for i in range(len(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_ids[i])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)
 
    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs, print_summary=False)
    ar = recalls.mean(axis=1)
    return ar
 
 
def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]
 
 
def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results
 
 
def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                json_results.append(data)
    return json_results
 
 
def segm2json(dataset, results):
    bbox_json_results = []
    segm_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det, seg = results[idx]
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
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
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results
 
 
def results2json(dataset, results, out_file):
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files