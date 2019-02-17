import numpy as np
from model.Mutils.submodules_multi_gpu import *
from model.Mutils import submodules_multi_gpu


class HOURGLASSYOLONet(object):

    def __init__(self, train_eval_visual='train'):
        self.num_anchors = cfg.NUM_ANCHORS
        self.yolo_version = cfg.YOLO_VERSION
        self.train_eval_visual = train_eval_visual
        self.l2 = tf.contrib.layers.l2_regularizer(cfg.L2_FACTOR) if cfg.L2 else None
        self.focal_loss = cfg.BOX_FOCAL_LOSS
        self.coord_sigmoid = cfg.COORD_SIGMOID
        self.wh_sigmoid = cfg.WH_SIGMOID
        self.r_object = cfg.R_OBJECT
        self.loss_factor = cfg.LOSS_FACTOR
        self.nMoudel = cfg.NUM_MOUDEL  # hourglass 中residual 模块的数量
        self.nStack = cfg.NUM_STACK  # hourglass 堆叠的层数
        self.nFeats = cfg.NUM_FEATS  # hourglass 中特征图的数量
        self.add_yolo_position = cfg.ADD_YOLO_POSITION
        self.num_class = len(cfg.COCO_CLASSES)
        self.image_size = cfg.IMAGE_SIZE
        self.nPoints = cfg.COCO_NPOINTS
        self.hg_cell_size = cfg.IMAGE_SIZE // 4
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        if self.yolo_version == '2':
            self.ch_size = self.num_anchors * 5
        else:
            self.ch_size = (self.num_class + self.boxes_per_cell * 5) if self.num_class != 1 \
                else self.boxes_per_cell * 5
        self.boundary1 = self.num_class if self.num_class != 1 else 0
        self.boundary2 = self.boundary1 + self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
        # self.batch_size = 1 if train_eval_visual == 'visual' else cfg.COCO_BATCH_SIZE
        if self.train_eval_visual == 'train':
            self.batch_size = cfg.COCO_BATCH_SIZE
        elif self.train_eval_visual == 'eval':
            self.batch_size = None
        else:
            self.batch_size = 1
        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')

    def build_network(self, x):
        # conv=conv2(input_x,7,[1,2,2,1])
        with tf.variable_scope('conv_pad3'):
            cp = pad_conv2(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 7, [1, 2, 2, 1], 3, 64, self.l2)
        with tf.variable_scope('batch_norm_relu'):
            bn = batch_norm_relu(cp)
        with tf.variable_scope('residual1'):
            r1 = bottleneck_residual(bn, [1, 1, 1, 1], 64, 128, self.l2)
        with tf.variable_scope('down_sampling'):
            ds = down_sampling(r1, [1, 2, 2, 1], [1, 2, 2, 1])
        with tf.variable_scope('residual2'):
            r2 = bottleneck_residual(ds, [1, 1, 1, 1], 128, 128, self.l2)
        with tf.variable_scope('residual3'):
            r3 = bottleneck_residual(r2, [1, 1, 1, 1], 128, self.nFeats, self.l2)

        output, yolo_output = None, None
        # hourglass 的输入
        h_input = r3
        for n in range(self.nStack):
            with tf.variable_scope('nStack' + str(n + 1)):
                with tf.variable_scope('hourglass' + str(n + 1)):
                    h1 = hourglass(h_input, self.nFeats, self.nMoudel, 4, self.l2)
                residual = h1
                for i in range(self.nMoudel):
                    with tf.variable_scope('residual' + str(i + 1)):
                        residual = bottleneck_residual(residual, [1, 1, 1, 1], self.nFeats, self.nFeats, self.l2)
                with tf.variable_scope('lin'):
                    r_lin = lin(residual, self.nFeats, self.nFeats, self.l2)

                    # add yolo_head in the tail of hg_net
                    if n == self.nStack - 1:
                        with tf.variable_scope('yolo'):
                            yolo_output = getattr(submodules_multi_gpu,
                                                  self.add_yolo_position)(r_lin,
                                                                          (self.ch_size, self.cell_size),
                                                                          self.l2)
                with tf.variable_scope('conv_same'):
                    output = conv2(r_lin, 1, [1, 1, 1, 1], self.nFeats, self.nPoints, self.l2, padding='VALID')  # 特征图输出
                if n < (self.nStack - 1):
                    # print(n)
                    with tf.variable_scope('next_input'):
                        # 卷积的输出
                        c_output = conv2(output, 1, [1, 1, 1, 1], self.nPoints, self.nFeats, self.l2)
                        h_input = tf.add(h_input, tf.add(r_lin, c_output))
        # transpose和reshape结果是不一样的
        output = tf.transpose(output, [0, 3, 1, 2], name='output')

        return output, yolo_output

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
          scope: variable_scope
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

    def loss_layer(self, predict, labels, scope):
        hg_logits, predicts = predict
        labels_det, labels_kp = labels
        with tf.variable_scope('loss_layer'):

            # probability of every class
            predict_classes = predicts[:, :, :, :self.boundary1] if self.num_class != 1 else None
            # object probability
            predict_scales = predicts[:, :, :, self.boundary1:self.boundary2]
            # x, y, w, h
            # x y: offset upon grids of cell_size X cell_size
            # w h: sqrt of scales(0~1) about w and h relative to pictures of 256*256
            #      for example real w h: (0.4, 0.8), so w h: (0.63, 0.89)
            predict_boxes_base = tf.reshape(
                predicts[:, :, :, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
            if self.coord_sigmoid and self.wh_sigmoid:
                # predict_boxes_xy = tf.sigmoid(predict_boxes_base[..., 0:2])
                # predict_boxes = tf.concat([predict_boxes_xy, predict_boxes_base[..., 2:]], 4)
                predict_boxes = tf.sigmoid(predict_boxes_base)
            elif self.coord_sigmoid:
                predict_boxes_xy = tf.sigmoid(predict_boxes_base[..., 0:2])
                predict_boxes = tf.concat([predict_boxes_xy, predict_boxes_base[..., 2:]], 4)
            elif self.wh_sigmoid:
                predict_boxes_wh = tf.sigmoid(predict_boxes_base[..., 2:4])
                predict_boxes = tf.concat([predict_boxes_base[..., 0:2], predict_boxes_wh], 4)
            else:
                predict_boxes = predict_boxes_base

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
            noobject_mask = tf.cast(tf.equal(object_mask, 0.0), tf.float32)
            # noobject_mask = tf.ones_like(
            #     object_mask, dtype=tf.float32) - object_mask

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

            if not self.focal_loss:
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
            else:  # focal loss
                object_sigmoid = tf.sigmoid(predict_scales)
                noobject_sigmoid = tf.ones_like(
                    object_sigmoid, dtype=tf.float32) - object_sigmoid

                object_pow_r = tf.pow(noobject_sigmoid, self.r_object)
                noobject_pow_r = tf.pow(object_sigmoid, self.r_object)

                object_delta = object_mask * tf.log(
                    tf.clip_by_value(object_sigmoid, 1e-8, 1.0)) * object_pow_r * -1
                noobject_delta = noobject_mask * tf.log(
                    tf.clip_by_value(noobject_sigmoid, 1e-8, 1.0)) * noobject_pow_r * -1

                object_loss = tf.reduce_mean(
                    tf.reduce_sum(object_delta, axis=[1, 2, 3]),
                    name='object_loss') * self.object_scale
                noobject_loss = tf.reduce_mean(
                    tf.reduce_sum(noobject_delta, axis=[1, 2, 3]),
                    name='object_loss') * self.noobject_scale
            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            yolo_loss = class_loss + object_loss + noobject_loss + coord_loss
            tf.add_to_collection('losses', yolo_loss)

            area_mask = tf.cast(tf.greater_equal(labels_kp, 0.0), tf.float32)
            diff1 = tf.subtract(hg_logits, labels_kp) * area_mask
            hg_loss = tf.reduce_mean(tf.nn.l2_loss(diff1, name='l2loss')) * self.loss_factor
            tf.add_to_collection('losses', hg_loss)

            loss = tf.add_n(tf.get_collection('losses', scope))

        # summary for train and val
        loss_list = []
        if self.num_class != 1:
            loss_list.append(class_loss)
        loss_list.append(object_loss)
        loss_list.append(noobject_loss)
        loss_list.append(coord_loss)
        loss_list.append(yolo_loss)
        loss_list.append(hg_loss)
        loss_list.append(loss)

        return loss, hg_loss, yolo_loss, loss_list

    def loss_layer_v2(self, predict, labels, scope):
        hg_logits, predicts = predict
        labels_det, labels_kp = labels
        mask_det = slice_tensor(labels_det, 4)
        labels_det = slice_tensor(labels_det, 0, 3)

        mask_det = tf.cast(tf.reshape(mask_det, shape=(-1, self.cell_size, self.cell_size, self.num_anchors)), tf.bool)
        with tf.variable_scope('loss_layer'):

            with tf.name_scope('mask'):
                # label result shape [[x, y, w, h, class], [], ...]
                masked_label = tf.boolean_mask(labels_det, mask_det)
                masked_pred = tf.boolean_mask(predicts, mask_det)
                neg_masked_pred = tf.boolean_mask(predicts, tf.logical_not(mask_det))

            with tf.name_scope('pred'):
                masked_pred_xy = tf.sigmoid(slice_tensor(masked_pred, 0, 1))
                masked_pred_wh = tf.exp(slice_tensor(masked_pred, 2, 3))
                masked_pred_xywh = tf.concat([masked_pred_xy, masked_pred_wh], 1)

                masked_pred_o = tf.sigmoid(slice_tensor(masked_pred, 4))
                masked_pred_no_o = tf.sigmoid(slice_tensor(neg_masked_pred, 4))
                # masked_pred_c = tf.nn.softmax(slice_tensor(masked_pred, 5, -1))

            with tf.name_scope('lab'):
                masked_label_xy = slice_tensor(masked_label, 0, 1)
                masked_label_wh = slice_tensor(masked_label, 2, 3)
                masked_label_xywh = slice_tensor(masked_label, 0, 3)
                # masked_label_c = slice_tensor(masked_label, 4)
                # masked_label_c_vec = tf.reshape(tf.one_hot(tf.cast(masked_label_c, tf.int32), depth=N_CLASSES),
                #                                 shape=(-1, N_CLASSES))

            with tf.name_scope('merge'):
                with tf.name_scope('loss_xy'):
                    loss_xy = tf.reduce_sum(tf.square(masked_pred_xy - masked_label_xy)) * self.coord_scale
                with tf.name_scope('loss_wh'):
                    loss_wh = tf.reduce_sum(tf.square(masked_pred_wh - masked_label_wh)) * self.coord_scale
                with tf.name_scope('loss_obj'):
                    ious = self.calc_iou(masked_pred_xywh, masked_label_xywh)
                    # loss_obj = tf.reduce_sum(tf.square(masked_pred_o - 1)) * self.object_scale
                    loss_obj = tf.reduce_sum(tf.square(masked_pred_o - ious)) * self.object_scale
                with tf.name_scope('loss_no_obj'):
                    loss_no_obj = tf.reduce_sum(tf.square(masked_pred_no_o)) * self.noobject_scale
                # with tf.name_scope('loss_class'):
                #     loss_c = tf.reduce_sum(tf.square(masked_pred_c - masked_label_c_vec))

            yolo_loss = loss_xy + loss_wh + loss_obj + loss_no_obj
            tf.add_to_collection('losses', yolo_loss)

            area_mask = tf.cast(tf.greater_equal(labels_kp, 0.0), tf.float32)
            diff1 = tf.subtract(hg_logits, labels_kp) * area_mask
            hg_loss = tf.reduce_mean(tf.nn.l2_loss(diff1, name='l2loss')) * self.loss_factor
            tf.add_to_collection('losses', hg_loss)

            loss = tf.add_n(tf.get_collection('losses', scope))

        # summary for train and val
        loss_list = [loss_obj, loss_no_obj, loss_xy + loss_wh, yolo_loss, hg_loss, loss]
        # loss_list.append(loss_obj)
        # loss_list.append(loss_no_obj)
        # loss_list.append(loss_xy + loss_wh)
        # loss_list.append(yolo_loss)
        # loss_list.append(hg_loss)
        # loss_list.append(loss)

        return loss, hg_loss, yolo_loss, loss_list
