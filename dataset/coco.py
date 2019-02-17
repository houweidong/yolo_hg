import numpy as np
import utils.config as cfg
from dataset.Dutils import read_coco_tf
import math
from dataset.Dutils.gene_box_v2 import *


class Coco(object):
    factor_list = [2.65, 5.3, 8, 10.6, 16, 21.2, 32, 42.5, 5000]
    diff_list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    grid_level = np.array([2, 4, 6, 8, 16])

    def __init__(self):
        self.sess = None
        self.yolo_version = cfg.YOLO_VERSION
        self.num_anchors = cfg.NUM_ANCHORS
        self.anchors = read_anchors_file('./dataset/Dutils/anchor' + str(self.num_anchors) + '.txt')
        self.coco_batch_size = cfg.COCO_BATCH_SIZE * cfg.GPU_NUMBER
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.hg_cell_size = cfg.IMAGE_SIZE // 4
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.classes = cfg.COCO_CLASSES
        self.num_class = len(self.classes)

        self.box_hm = cfg.BOX_HOT_MAP
        # for yolo head
        self.box_hm_factor = Coco.factor_list[cfg.BOX_HOT_MAP_LEVEL]

        # for keypoints head
        self.hg_hm_diff_factor = Coco.diff_list[cfg.HG_HOT_MAP_DIFF_LEVEL]
        hg_factor_list = Coco.grid_level / math.sqrt(math.log(10)) / math.pow(math.pow(self.hg_cell_size / 2, 2),
                                                                              self.hg_hm_diff_factor)
        self.hg_hm_factor = hg_factor_list[cfg.HG_HOT_MAP_LEVEL]

        # self.phase = phase
        self.nPoints = cfg.COCO_NPOINTS
        self.coco_train_fn = cfg.COCO_TRAIN_FILENAME
        self.coco_val_fn = cfg.COCO_VAL_FILENAME
        self.coco_epoch_size = cfg.COCO_EPOCH_SIZE
        self.train_im_batch, \
            self.train_labels_det_batch, self.train_labels_category, \
            self.train_labels_kp_batch, self.train_num_points \
            = read_coco_tf.batch_samples_all_categories(self.coco_batch_size,
                                                        self.coco_train_fn,
                                                        shuffle=False)
        self.val_im_batch, \
            self.val_labels_det_batch, self.val_labels_category, \
            self.val_labels_kp_batch, self.val_num_points \
            = read_coco_tf.batch_samples_all_categories(self.coco_batch_size,
                                                        self.coco_val_fn,
                                                        shuffle=False)

    def get(self, phase):
        if phase == "train":
            example, l_det, l_cg, l_kp, l_np = self.sess.run([self.train_im_batch,
                                                              self.train_labels_det_batch,
                                                              self.train_labels_category,
                                                              self.train_labels_kp_batch,
                                                              self.train_num_points])
        else:
            example, l_det, l_cg, l_kp, l_np = self.sess.run([self.val_im_batch,
                                                              self.val_labels_det_batch,
                                                              self.val_labels_category,
                                                              self.val_labels_kp_batch,
                                                              self.val_num_points])
        images = self.image_normalization(example)  # 归一化图像
        if self.yolo_version == '1':
            labels_det = self.batch_gene_hm_bbox(l_det, l_cg)
        else:
            labels_det = self.batch_gene_box_v2(l_det, l_cg)
        labels_kp = self.batch_gene_hm_kp(l_kp, l_det, l_np, l_cg)  # heatmap label

        return images, labels_det, labels_kp

    def gene_hm_box(self, xmin, xmax, ymin, ymax, label):
        l, r, t, d = map(lambda x: x * self.cell_size / self.image_size, (xmin, xmax, ymin, ymax))
        center_col, center_row = math.modf(l)[0] + (r - l) / 2 - 0.5, math.modf(t)[0] + (d - t) / 2 - 0.5
        l, t = math.floor(l), math.floor(t)
        r, d = math.ceil(r), math.ceil(d)
        grid_w, grid_h = r - l, d - t
        imag_w, imag_h = xmax - xmin, ymax - ymin
        # consider image area
        factor = 1 / np.power(imag_h * imag_w, 1 / 2) * self.box_hm_factor
        sigma_w, sigma_h = factor * imag_w / 2, factor * imag_h / 2
        col = np.reshape(np.array([np.arange(grid_w)] * grid_h), (grid_h, grid_w))
        row = np.transpose(np.reshape(np.array([np.arange(grid_h)] * grid_w), (grid_w, grid_h)))
        prob = np.exp(-1 * ((np.square(col - center_col)) / sigma_w ** 2
                            + (np.square(row - center_row)) / sigma_h ** 2))

        boxes = np.array([(xmax + xmin) / 2.0, (ymax + ymin) / 2.0, xmax - xmin, ymax - ymin])
        condition = label[t:d, l:r, 0] > prob
        label[t:d, l:r, 0] = np.where(condition, label[t:d, l:r, 0], prob)
        label[t:d, l:r, 1:5] = np.where(condition[:, :, np.newaxis],
                                        label[t:d, l:r, 1:5],
                                        boxes[np.newaxis, np.newaxis, :])

    def batch_gene_hm_bbox(self, batch_det, batch_cg):

        label_ch = self.num_class + 5
        if self.num_class == 1:
            label_ch = 5
        labels = np.zeros(
            (self.coco_batch_size, self.cell_size, self.cell_size, label_ch), dtype=np.float32)

        for i in range(self.coco_batch_size):
            l_det, l_cg = batch_det[i], batch_cg[i]
            l_det = np.reshape(l_det, (-1, 4))
            label = np.zeros((self.cell_size, self.cell_size, label_ch), dtype=np.float32)
            for obj, cg in zip(l_det, l_cg):
                if np.array_equal(obj, [256, 0, 256, 0]) or np.array_equal(obj, [0, 0, 0, 0]):
                    continue
                if self.num_class == 1 and cg != 1:
                    continue
                x1 = max(min(obj[0], self.image_size), 0)
                y1 = max(min(obj[1], self.image_size), 0)
                x2 = max(min(obj[2], self.image_size), 0)
                y2 = max(min(obj[3], self.image_size), 0)
                boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
                x_ind = int(boxes[0] * self.cell_size / self.image_size)
                y_ind = int(boxes[1] * self.cell_size / self.image_size)

                if self.box_hm:
                    self.gene_hm_box(x1, x2, y1, y2, label)
                else:
                    if label[y_ind, x_ind, 0] == 1:
                        continue
                    label[y_ind, x_ind, 0] = 1
                    label[y_ind, x_ind, 1:5] = boxes

                # multi classification
                if self.num_class != 1:
                    label[y_ind, x_ind, 5 + cg] = 1
            labels[i, :, :, :] = label

        return labels

    def batch_gene_box_v2(self, batch_det, batch_cg):
        # TODO  not support multi class, just support person, the num_class always is 1
        labels = np.zeros(
            [self.coco_batch_size, self.cell_size, self.cell_size, self.num_anchors, 5], dtype=np.float32)

        for i in range(self.coco_batch_size):
            l_det, l_cg = batch_det[i], batch_cg[i]
            l_det = np.reshape(l_det, (-1, 4))
            label = np.zeros([self.cell_size, self.cell_size, self.num_anchors, 5], dtype=np.float32)
            for obj, cg in zip(l_det, l_cg):
                if np.array_equal(obj, [256, 0, 256, 0]) or np.array_equal(obj, [0, 0, 0, 0]):
                    continue
                if cg != 1:
                    continue
                x1 = max(min(obj[0], self.image_size), 0)
                y1 = max(min(obj[1], self.image_size), 0)
                x2 = max(min(obj[2], self.image_size), 0)
                y2 = max(min(obj[3], self.image_size), 0)
                roi = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
                # roi = [x1, y1, x2 - x1, y2 - y1]
                active_indxs = get_active_anchors(roi, self.anchors)
                grid_x, grid_y = get_grid_cell(roi, self.image_size, self.image_size, self.cell_size, self.cell_size)

                for active_indx in active_indxs:
                    anchor_label = roi2label(roi, self.anchors[active_indx],
                                             self.image_size, self.image_size,
                                             self.cell_size, self.cell_size)
                    label[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [1.0]))
            labels[i, :, :, :] = label

        return labels

    def gene_hm_kp(self, kps_resize, area):
        hm_kps = np.zeros((self.nPoints, self.hg_cell_size, self.hg_cell_size), dtype=np.float32)
        sigma_diff = area ** self.hg_hm_diff_factor
        sigma = sigma_diff * self.hg_hm_factor
        for i in range(self.nPoints):
            kp = kps_resize[i]
            if np.array_equal(kp, [0, 0]) or np.array_equal(kp, [self.image_size / 4, 0]):
                continue
            else:
                col = np.reshape(np.array([np.arange(self.hg_cell_size)] * self.hg_cell_size),
                                 (self.hg_cell_size, self.hg_cell_size))
                row = np.transpose(np.reshape(np.array([np.arange(self.hg_cell_size)] * self.hg_cell_size),
                                              (self.hg_cell_size, self.hg_cell_size)))
                center_col, center_row = kp
                hm_kps[i] = np.exp(-1 * (np.square(col - center_col)
                                         + np.square(row - center_row)) / sigma ** 2)

        return hm_kps

    def batch_gene_hm_kp(self, batch_kp, batch_bbox, batch_points_num, batch_id, if_gauss=True):
        # has a little error, need use original image size(not self.image_size) to translate the coord
        # right and down slightly.
        # formula: 1 / (origin_ims / cell_size) / 2
        batch_kp_resize = batch_kp * self.hg_cell_size / self.image_size
        batch_hm_kp = np.zeros((self.coco_batch_size, self.nPoints, self.hg_cell_size, self.hg_cell_size),
                               dtype=np.float32)
        batch_bbox = np.reshape(batch_bbox, (self.coco_batch_size, -1, 4))
        if if_gauss:
            for i in range(self.coco_batch_size):
                points_nums = batch_points_num[i]
                bbox = batch_bbox[i]
                ids = batch_id[i]
                for j in range(cfg.COCO_MAX_OBJECT_PER_PIC):
                    points_num = points_nums[j]
                    category = ids[j]
                    box = bbox[j]
                    area = (box[2] - box[0]) * self.hg_cell_size / self.image_size * \
                           (box[3] - box[1]) * self.hg_cell_size / self.image_size
                    box = box * self.hg_cell_size / self.image_size
                    if category == 1 and not points_num:
                        no_points_bbox = box
                        x1 = math.floor(max(min(no_points_bbox[0], self.hg_cell_size), 0.0))
                        y1 = math.floor(max(min(no_points_bbox[1], self.hg_cell_size), 0.0))
                        x2 = math.ceil(max(min(no_points_bbox[2], self.hg_cell_size), 0.0))
                        y2 = math.ceil(max(min(no_points_bbox[3], self.hg_cell_size), 0.0))
                        batch_hm_kp[i, :, y1:y2, x1:x2] += -0.1
                    elif points_num:
                        prob = self.gene_hm_kp(
                            batch_kp_resize[i][j * self.nPoints:j * self.nPoints + self.nPoints], area)

                        condition = batch_hm_kp[i] > prob
                        batch_hm_kp[i] = np.where(condition, batch_hm_kp[i], prob)

                    else:
                        continue
        else:
            # TODO SUPPORT SINGLE POINT FUTURE
            print('single point not support now')
            pass
        return batch_hm_kp

    @staticmethod
    def image_normalization(image):
        # 除以255对图像进行归一化
        nor_image = np.asarray(image, dtype=np.float32)
        for i in range(image.shape[0]):
            nor_image[i] = image[i] / 255.0
        return nor_image

# import matplotlib.pyplot as plt
# from PIL import Image
# from dataset.Dutils.read_coco_tf import *
# import utils.config as cfg
#
# cfg.HG_HOT_MAP_DIFF_LEVEL = 2
# cfg.HG_HOT_MAP_LEVEL = 2
# with tf.Session() as sess:  # 开始一个会话
#     coco = Coco()
#     coco.coco_batch_size = 1
#     coco.box_hm = True
#     coco.box_hm_factor = Coco.factor_list[2]
#     init_op = tf.global_variables_initializer()
#     tf.local_variables_initializer().run()
#     sess.run(init_op)
#     batch_size = 1
#
#     b_image, b_bbox, b_category_id, b_label, b_num_kpoints = \
#         batch_samples_all_categories(batch_size,
#                                      '/root/dataset/tfrecord1/val',
#                                      False)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(7900):
#         try:
#             # 在会话中取出image和label
#             r_image, r_label, r_bbox, r_num_kpoints, r_category_id = sess.run(
#                 [b_image, b_label, b_bbox, b_num_kpoints, b_category_id])
#             r_bbox_resize = r_bbox * 64 / 256
#         except tf.errors.OutOfRangeError as info:
#             print('info', info)
#             exit()
#         else:
#             print(i)
#             # print(r_category_id)
#             hm_result = coco.batch_gene_hm_kp(r_label, r_bbox, r_num_kpoints, r_category_id)
#             det_result = coco.batch_gene_box_v2(r_bbox, r_category_id)
#             # det_result = np.transpose(det_result, [0, 3, 1, 2])[0][0]
#             det_result = det_result[0]
#
#             for j in range(batch_size):
#                 or_x = r_label[j][:, 0] * 64 / 256
#                 or_y = r_label[j][:, 1] * 64 / 256
#
#                 # or_x = r_label[j][0::17][:, 0] * 64 / 256
#                 # or_y = r_label[j][0::17][:, 1] * 64 / 256
#                 print(or_x)
#                 print(or_y)
#
#                 img = Image.fromarray(r_image[j])
#                 plt.imshow(img, cmap='Greys_r')
#                 for iii in range(7):
#                     plt.matshow(det_result[:, :, iii, 4])
#                 # plt.matshow(np.sum(hm_result[j], axis=0))
#                 plt.matshow((hm_result[j][0]))
#
#                 all_x = []
#                 all_y = []
#                 # for index in range(1):
#                 #     position=np.argmax(hm_result[j][index])
#                 #     y,x=divmod(position, 64)
#                 #     all_x.append(x)
#                 #     all_y.append(y)
#                 #     plt.matshow(hm_result[j][index])
#                 plt.plot(all_x, all_y, 'r+')
#                 plt.plot(or_x, or_y, 'g+')
#                 leng = r_bbox.shape[1]
#                 # rect = plt.Rectangle((-0.45, -0.45), 30, 30,
#                 #                      linewidth=1,
#                 #                      edgecolor='g',
#                 #                      facecolor='none')
#                 #
#                 # plt.gca().add_patch(rect)
#                 for ii in range(leng // 2):
#                     rect = plt.Rectangle((r_bbox_resize[j][ii * 2][0] - 0.5, r_bbox_resize[j][ii * 2][1] - 0.5),
#                                          r_bbox_resize[j][ii * 2 + 1][0] - r_bbox_resize[j][ii * 2][0],
#                                          r_bbox_resize[j][ii * 2 + 1][1] - r_bbox_resize[j][ii * 2][1],
#                                          linewidth=1,
#                                          edgecolor='g',
#                                          facecolor='none')
#
#                     plt.gca().add_patch(rect)
#                 plt.show()
#     coord.request_stop()
#     coord.join(threads)
