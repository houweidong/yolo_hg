import numpy as np
import tensorflow as tf
from utils import config as cfg
from model.resnet_model import model

slim = tf.contrib.slim


class HOURGLASSYOLONet(object):

    def __init__(self):
        # , is_training=True):
        # self.lf_tsp = cfg.LOOK_FEATURES_TRANSPOSE
        self.loss_factor = cfg.LOSS_FACTOR
        self.is_training = True
        self.add_yolo_position = cfg.ADD_YOLO_POSITION
        self.num_class = len(cfg.COCO_CLASSES)
        self.image_size = cfg.IMAGE_SIZE
        self.nPoints = cfg.COCO_NPOINTS
        self.hg_cell_size = cfg.HG_CELL_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        # self.output_size = (self.cell_size * self.cell_size) *\
        #     (self.num_class + self.boxes_per_cell * 5) if self.num_class != 1 \
        #     else (self.cell_size * self.cell_size) * self.boxes_per_cell * 5
        self.ch_size = (self.num_class + self.boxes_per_cell * 5) if self.num_class != 1 \
            else self.boxes_per_cell * 5
        # self.scale = 1.0 * self.image_size / self.cell_size
        # self.boundary1 = self.cell_size * self.cell_size * self.num_class if self.num_class != 1 else 0
        self.boundary1 = self.num_class if self.num_class != 1 else 0
        self.boundary2 = self.boundary1 + self.boxes_per_cell
        # self.boundary2 = self.boundary1 +\
        #     self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        # self.batch_size = cfg.BATCH_SIZE
        self.batch_size = cfg.COCO_BATCH_SIZE
        # self.keep_prob = cfg.KEEP_PROB
        # self.alpha = cfg.ALPHA

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        self.hg_logits, self.yolo_logits = self.build_network()
        # if self.is_training:
        self.labels_det = tf.placeholder(
            tf.float32,
            [None,
             self.cell_size,
             self.cell_size,
             self.num_class + 5 if self.num_class != 1 else 5])
        self.labels_kp = tf.placeholder(
            tf.float32,
            [None,
             self.nPoints,
             self.hg_cell_size,
             self.hg_cell_size])
        self.loss, self.hg_loss, self.yolo_loss = \
            self.loss_layer([self.hg_logits, self.yolo_logits],
                            [self.labels_det, self.labels_kp])

    def build_network(self):
        return model(self.images,
                     (self.batch_size, self.cell_size, self.cell_size, self.ch_size),
                     # self.alpha,
                     # self.keep_prob,
                     # self.is_training,
                     self.add_yolo_position)

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predict, labels, scope='loss_layer'):
        hg_logits, predicts = predict
        labels_det, labels_kp = labels
        with tf.variable_scope(scope):

            # probability of every class
            predict_classes = predicts[:, :, :, self.boundary1] if self.num_class != 1 else None
            # object probability
            predict_scales = predicts[:, :, :, self.boundary1:self.boundary2]
            # x, y, w, h
            # x y: offset upon grids of cell_size X cell_size
            # w h: sqrt of scales(0~1) about w and h relative to pictures of 256*256
            #      for example real w h: (0.4, 0.8), so w h: (0.63, 0.89)
            predict_boxes = tf.reshape(
                    predicts[:, :, :, self.boundary2:],
                    [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            # object indicator(1 represents has object, 0 no object)
            response = tf.reshape(
                labels_det[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            # x, y, w, h
            # x y: location of objects relative to pictures of 256*256, for example (200, 136)
            # w h: width and height of the objects relative to pictures of 256*256, for example (50, 64)
            boxes = tf.reshape(
                labels_det[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            # copy boxes according to per_cell value for every cell, and transpose x y w h to scale(0~1)
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size

            # classification of objects if has more than one class
            classes = labels_det[..., 5:] if self.num_class != 1 else None

            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))

            # transpose x y to scale(0~1) according to offset and cell_size, and caculate scale of w h
            # according to sqrt of w, h
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keepdims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss
            # class_loss = None
            class_loss = tf.constant(0.0)
            if self.num_class != 1:
                class_delta = response * (predict_classes - classes)
                class_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                    name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale
            yolo_loss = class_loss + object_loss + noobject_loss + coord_loss

            diff1 = tf.subtract(hg_logits, labels_kp)
            hg_loss = tf.reduce_mean(tf.nn.l2_loss(diff1, name='l2loss')) * self.loss_factor
            # tf.losses.add_loss(hg_loss)
            loss = yolo_loss + hg_loss

            # summary for train and val
        name_lt = ['train', 'val']
        for name in name_lt:
            if self.num_class != 1:
                tf.summary.scalar(name + '/yolo/class_loss', class_loss, collections=[name])
            tf.summary.scalar(name + '/yolo/object_loss', object_loss, collections=[name])
            tf.summary.scalar(name + '/yolo/noobject_loss', noobject_loss, collections=[name])
            tf.summary.scalar(name + '/yolo/coord_loss', coord_loss, collections=[name])
            tf.summary.scalar(name + '/yolo/yolo_loss', yolo_loss, collections=[name])
            tf.summary.histogram(name + '/yolo/iou', iou_predict_truth, collections=[name])
            tf.summary.scalar(name + '/hg_loss', hg_loss, collections=[name])
            tf.summary.scalar(name + '/total_loss', loss, collections=[name])
        # tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0], collections='train')
        # tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1], collections='train')
        # tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2], collections='train')
        # tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3], collections='train')
        return loss, hg_loss, yolo_loss
