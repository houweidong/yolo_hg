from utils import config as cfg
import numpy as np
from sklearn.metrics import average_precision_score
from utils.timer import Timer

num_class = len(cfg.COCO_CLASSES)
boundary1 = num_class if num_class != 1 else 0
boundary2 = boundary1 + cfg.BOXES_PER_CELL

# def iou(box1, box2):
#     tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
#         max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
#     lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
#         max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
#     inter = 0 if tb < 0 or lr < 0 else tb * lr
#     return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)


def iou2gt(pred, gt):
    """whether a bbbox is required by looking iou with gt and class num
    Args:
      pred: tuple (boxes_filtered, classes_num_filtered)
            boxes_filtered: 2-D array [NUM_PRI_BBOX 4]  ====> (x_center, y_center, w, h)
      gt: tuple (boxes_filtered_gt, classes_num_filtered_gt)
          boxes_filtered_gt: 2-D array [NUM_GT_BBOX 4] ===> (x_center, y_center, w, h)
    Return:
      iou_indicator: 1-D array [NUM_PRI_BBOX]
      type: int(1 or 0)
    """
    boxes_filtered, classes_num_filtered = pred
    boxes_filtered_gt, classes_num_filtered_gt = gt
    boxes_repeat = np.repeat(boxes_filtered, boxes_filtered_gt.shape[0], axis=0)
    boxes_tile_gt = np.tile(boxes_filtered_gt, (boxes_filtered.shape[0], 1))

    # calculate iou
    tb = np.minimum(boxes_repeat[:, 0] + 0.5 * boxes_repeat[:, 2], boxes_tile_gt[:, 0] + 0.5 * boxes_tile_gt[:, 2]) - \
        np.maximum(boxes_repeat[:, 0] - 0.5 * boxes_repeat[:, 2], boxes_tile_gt[:, 0] - 0.5 * boxes_tile_gt[:, 2])
    lr = np.minimum(boxes_repeat[:, 1] + 0.5 * boxes_repeat[:, 3], boxes_tile_gt[:, 1] + 0.5 * boxes_tile_gt[:, 3]) - \
        np.maximum(boxes_repeat[:, 1] - 0.5 * boxes_repeat[:, 3], boxes_tile_gt[:, 1] - 0.5 * boxes_tile_gt[:, 3])

    inter = np.maximum(0, tb) * np.maximum(0, lr)
    iou = inter / (boxes_repeat[:, 2] * boxes_repeat[:, 3] + boxes_tile_gt[:, 2] * boxes_tile_gt[:, 3] - inter)
    mn = (iou > cfg.IOU_THRESHOLD).astype(np.int)
    indicator_bbox = np.max(np.reshape(mn,
                                       [boxes_filtered.shape[0],
                                        boxes_filtered_gt.shape[0]]),
                            axis=1)

    # the indicator_class only support one class object detection
    indicator_class = np.ones(indicator_bbox.shape)
    return np.minimum(indicator_bbox, indicator_class)


def py_cpu_nms(dets):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    thresh = cfg.IOU_THRESHOLD
    x1 = dets[:, 0] - (dets[:, 2] / 2)
    y1 = dets[:, 1] - (dets[:, 3] / 2)
    x2 = dets[:, 0] + (dets[:, 2] / 2)
    y2 = dets[:, 1] + (dets[:, 3] / 2)
    # 每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.arange(len(dets))
    keep = []
    # 保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]  # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
    return keep


def interpret_output(output, labels_det):
    """translate output of yolo to probs, iou_indecator for AP calculation
    Args:
        output: the yolo head of hg_yolo net
        labels_det: detection label
    Return:
        iou_indicator: 1-D array [NUM_PRI_BBOX]  type: int(1 or 0)
        probs_filtered: 1-D array [NUM_PRI_BBOX] type: float
                        bbox probs
    """
    probs = np.zeros((cfg.CELL_SIZE, cfg.CELL_SIZE,
                      cfg.BOXES_PER_CELL, num_class))
    class_probs = output[:, :, :boundary1] if num_class != 1 \
        else np.ones((cfg.CELL_SIZE, cfg.CELL_SIZE, 1))

    scales = output[:, :, boundary1:boundary2]
    boxes = np.reshape(
        output[:, :, boundary2:],
        (cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 4))

    offset = np.array(
        [np.arange(cfg.CELL_SIZE)] * cfg.CELL_SIZE * cfg.BOXES_PER_CELL)
    offset = np.transpose(
        np.reshape(
            offset,
            [cfg.BOXES_PER_CELL, cfg.CELL_SIZE, cfg.CELL_SIZE]),
        (1, 2, 0))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / cfg.CELL_SIZE
    boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

    boxes *= cfg.IMAGE_SIZE

    for i in range(cfg.BOXES_PER_CELL):
        for j in range(num_class):
            probs[:, :, i, j] = np.multiply(
                class_probs[:, :, j], scales[:, :, i])

    probs = np.clip(probs, 0, 1)
    filter_mat_probs = np.array(probs >= 0.0, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)

    # [[bbox1] [bbox2] ...]
    boxes_filtered = boxes[filter_mat_boxes[0],
                           filter_mat_boxes[1], filter_mat_boxes[2]]
    # [0.7, 0.5, 0.91, ...]
    probs_filtered = probs[filter_mat_probs]
    # [2, 3, 0, 20, ...]
    classes_num_filtered = np.argmax(
        probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

    # [2, 1, 0, ...]
    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    # NMS
    keep = py_cpu_nms(boxes_filtered)

    # select bbox should keep
    boxes_filtered = boxes_filtered[keep]
    probs_filtered = probs_filtered[keep]
    classes_num_filtered = classes_num_filtered[keep]

    # object indicator(1 represents has object, 0 no object)
    response = labels_det[..., 0]
    # x, y, w, h
    # x y: location of objects relative to pictures of 256*256, for example (200, 136)
    # w h: width and height of the objects relative to pictures of 256*256, for example (50, 64)
    boxes_gt = labels_det[..., 1:5]

    # classification of objects if has more than one class
    classes_gt = labels_det[..., 5:] if num_class != 1 \
        else np.zeros([cfg.CELL_SIZE, cfg.CELL_SIZE, 1])

    filter_mat_response = np.array(response == 1, dtype='bool')
    filter_mat_gt = np.nonzero(filter_mat_response)
    # print(filter_mat_gt.shape[0])
    # print(filter_mat_gt.shape[1])
    boxes_filtered_gt = boxes_gt[filter_mat_gt[0], filter_mat_gt[1]]
    classes_num_filtered_gt = np.argmax(classes_gt, 2)[filter_mat_gt[0], filter_mat_gt[1]]
    iou_indicator = iou2gt((boxes_filtered, classes_num_filtered),
                           (boxes_filtered_gt, classes_num_filtered_gt))
    return iou_indicator, probs_filtered


def ev_ap(net_output, val_det_bt):
    # calculate ap of a batch
    ap = 0
    for i in range(net_output.shape[0]):
        lb, prob = interpret_output(net_output[i], val_det_bt[i])
        if not np.nonzero(lb)[0]:
            continue
        ap += average_precision_score(y_true=lb, y_score=prob)
    return ap / cfg.COCO_BATCH_SIZE

