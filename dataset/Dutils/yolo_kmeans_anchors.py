# coding=utf-8
# k-means ++ for YOLOv2 anchors
# 通过k-means ++ 算法获取YOLOv2需要的anchors的尺寸
import numpy as np
import json
import os
import xml.etree.ElementTree as EmTr
from utils.logger import Logger


# 定义Box类，描述bounding box的坐标
class Box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


# 计算两个box在某个轴上的重叠部分
# x1是box1的中心在该轴上的坐标
# len1是box1在该轴上的长度
# x2是box2的中心在该轴上的坐标
# len2是box2在该轴上的长度
# 返回值是该轴上重叠的长度
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


# 计算box a 和box b 的交集面积
# a和b都是Box类型实例
# 返回值area是box a 和box b 的交集面积
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# 计算 box a 和 box b 的并集面积
# a和b都是Box类型实例
# 返回值u是box a 和box b 的并集面积
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# 计算 box a 和 box b 的 iou
# a和b都是Box类型实例
# 返回值是box a 和box b 的iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


# 使用k-means ++ 初始化 centroids，减少随机初始化的centroids对最终结果的影响
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# 返回值centroids 是初始化的n_anchors个centroid
def init_centroids(boxes, n_anchors):
    centroids = []
    boxes_num = len(boxes)

    centroid_index = np.random.choice(boxes_num, 1)
    centroids.append(boxes[centroid_index[0]])

    print(centroids[0].w, centroids[0].h)

    for centroid_index in range(0, n_anchors - 1):

        sum_distance = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - box_iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance * np.random.random()

        for i in range(0, boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break

    return centroids


# 进行 k-means 计算新的centroids
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# centroids是所有簇的中心
# 返回值new_centroids 是计算出的新簇中心
# 返回值groups是n_anchors个簇包含的boxes的列表
# 返回值loss是所有box距离所属的最近的centroid的距离的和
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss


# 计算给定bounding boxes的n_anchors数量的centroids
# label_path是训练集列表文件地址
# n_anchors 是anchors的数量
# loss_convergence是允许的loss的最小变化值
# grid_size * grid_size 是栅格数量
# iterations_num是最大迭代次数
# plus = 1时启用k means ++ 初始化centroids
def compute_centroids(label_path, n_anchors, loss_convergence, grid_size, iterations_num, plus, name='coco'):
    log = Logger.get_logger('anchor' + str(n_ac) + '.txt')
    boxes = []
    if name == "coco":
        with open(label_path, 'r') as label:
            groundtruth_data = json.load(label)
            image_wh = {}
            for img in groundtruth_data['images']:
                image_wh[img['id']] = [img['width'], img['height']]
            for annotation in groundtruth_data['annotations']:
                if annotation['category_id'] != 1:
                    continue
                image_id = annotation['image_id']
                if annotation['iscrowd'] == 1:
                    continue
                boxes.append(Box(0, 0, float(annotation['bbox'][2]) / image_wh[image_id][0],
                                 float(annotation['bbox'][3]) / image_wh[image_id][1]))
    elif name == 'pascal':
        classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                   'train', 'tvmonitor']
        class_to_ind = dict(zip(classes, range(len(classes))))
        for file in os.listdir(label_path):
            filename = os.path.join(label_path, file)
            tree = EmTr.parse(filename)
            objs = tree.findall('object')
            size = tree.find('size')
            for obj in objs:
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                cls_ind = class_to_ind[obj.find('name').text.lower().strip()]
                # person -> 14
                if cls_ind != 14:
                    continue
                x1 = float(bbox.find('xmin').text) - 0.5
                y1 = float(bbox.find('ymin').text) - 0.5
                x2 = float(bbox.find('xmax').text) - 0.5
                y2 = float(bbox.find('ymax').text) - 0.5
                w, h = float(size.find('width').text), float(size.find('height').text)
                if x1 < 0 or y1 < 0:
                    print('<0')
                if x2 > w or y2 > h:
                    print('>weight or >height')
                boxes.append(Box(0, 0, (x2 - x1) / w, (y2 - y1) / h))

    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    down_time = 0
    loss = float("inf")
    while iterations <= iterations_num:
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations = iterations + 1
        print("loss = %f" % loss)
        if old_loss - loss < 0:
            down_time += 1
        else:
            down_time = 0
        if abs(old_loss - loss) < loss_convergence or down_time == 1:
            break
        old_loss = loss
        # for centroid in centroids:
        #     print(centroid.w * grid_size, centroid.h * grid_size)

    # print result
    acc = (1 - (loss / len(boxes))) * 100
    if acc > globals()['acc' + str(n_anchors)]:
        print("good {} result:".format(name))
        for centroid in centroids:
            log.info('{} {}'.format(centroid.w * grid_size, centroid.h * grid_size))
        log.info("Len: {}".format(len(boxes)))
        log.info("Accuracy: {:.2f}%".format(acc))
        globals()['acc' + str(n_anchors)] = acc
    else:
        print('acc lower, continue')


nm = 'coco'
if nm == 'coco':
    lb_pt = '/root/dataset/annotations_trainval2017/annotations/instances_train2017.json'
else:
    lb_pt = '/root/dataset/data/pascal_voc/VOCdevkit/VOC2012/Annotations'
# n_ac = 9
loss_cvg = 1e-6
gs = 1
it_num = 1000
pl = 1

tm_num = 30
# init
for n_ac in [7, 10, 13]:
    Logger('anchor' + str(n_ac) + '.txt', level='debug')
    locals()['acc' + str(n_ac)] = 0.0

for _ in range(tm_num):
    # list_nac.extend(list_base)
    for n_ac in [7, 10, 13]:
        compute_centroids(lb_pt, n_ac, loss_cvg, gs, it_num, pl, nm)
