import utils.config as cfg
import numpy as np


def read_anchors_file(file_path):
    anchors = []
    with open(file_path, 'r') as file:
        for line in file.read().splitlines():
            anchors.append(list(map(float, line.split())))

    return np.array(anchors)


def iou_wh(r1, r2):
    min_w = min(r1[0], r2[0])
    min_h = min(r1[1], r2[1])
    area_r1 = r1[0] * r1[1]
    area_r2 = r2[0] * r2[1]

    intersect = min_w * min_h
    union = area_r1 + area_r2 - intersect

    return intersect / union


def get_grid_cell(roi, raw_w, raw_h, grid_w, grid_h):
    # x_center = roi[0] + roi[2] / 2.0
    # y_center = roi[1] + roi[3] / 2.0
    x_center = roi[0]
    y_center = roi[1]

    grid_x = int(x_center / float(raw_w) * float(grid_w))
    grid_y = int(y_center / float(raw_h) * float(grid_h))

    return grid_x, grid_y


def get_active_anchors(roi, anchors):
    indxs = []
    iou_max, index_max = 0, 0
    for i, a in enumerate(anchors):
        a = a * float(cfg.IMAGE_SIZE)
        iou = iou_wh(roi[2:], a)
        if iou > cfg.IOU_THRESHOLD_V2:
            indxs.append(i)
        if iou > iou_max:
            iou_max, index_max = iou, i

    if len(indxs) == 0:
        indxs.append(index_max)

    return indxs


def roi2label(roi, anchor, raw_w, raw_h, grid_w, grid_h):
    # x_center = roi[0] + roi[2] / 2.0
    # y_center = roi[1] + roi[3] / 2.0
    x_center = roi[0]
    y_center = roi[1]

    grid_x = x_center / float(raw_w) * float(grid_w)
    grid_y = y_center / float(raw_h) * float(grid_h)

    grid_x_offset = grid_x - int(grid_x)
    grid_y_offset = grid_y - int(grid_y)

    roi_w_scale = roi[2] / float(raw_w) / anchor[0]
    roi_h_scale = roi[3] / float(raw_h) / anchor[1]

    label = [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale]

    return label


def onehot(idx, num):
    ret = np.zeros([num], dtype=np.float32)
    ret[idx] = 1.0

    return ret

# a = read_anchors_file('./anchor7.txt')
# for i, ac in enumerate(a):
#     print(ac)
