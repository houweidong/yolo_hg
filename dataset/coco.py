import numpy as np
from utils import config as cfg
from dataset.Dutils import read_coco_tf, gene_hm, processing


# import matplotlib.pyplot as plt


class Coco(object):
    def __init__(self):
        self.sess = None
        self.box_hm = cfg.BOX_HOT_MAP
        # self.box_hm_level = cfg.BOX_HOT_MAP_LEVEL
        # self.box_hm_gaussian = cfg.BOX_HOT_MAP_GAUSSIAN
        # self.box_hm_sigma = cfg.BOX_HOT_MAP_SIGMA
        # self.box_hm_prob = self.prepare_prob()
        # self.box_hm_prob_size = 2 * self.box_hm_level + 1
        self.coco_batch_size = cfg.COCO_BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.classes = cfg.COCO_CLASSES
        self.num_class = len(self.classes)
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

    # def prepare_prob(self):
    #     if self.box_hm:
    #         prob_all = []
    #         for level in range(self.cell_size // 2):
    #             # dmax = self.box_hm_level * pow(2, 0.5)
    #             box_hm_prob_size = 2 * level + 1
    #             col = np.reshape(np.array([np.arange(box_hm_prob_size)] * box_hm_prob_size),
    #                              (box_hm_prob_size, box_hm_prob_size))
    #             row = np.transpose(col)
    #             center = level
    #             # print(d)
    #             # if not self.box_hm_gaussian:
    #             #     d = np.sqrt(np.square(row - center) + np.square(col - center))
    #             #     prob = np.ones((self.box_hm_prob_size, self.box_hm_prob_size)) - 0.9 / dmax * d
    #             # else:
    #             prob = np.exp(-1 * (np.square(row - center) +
    #                                 np.square(col - center)) / self.box_hm_sigma ** 2)
    #             prob_all.append(prob)
    #         return prob_all
    #     else:
    #         return None

    # def gn_hm_prob(self):
    #     prob_all = []
    #     for level in range(self.cell_size // 2):
    #         # dmax = self.box_hm_level * pow(2, 0.5)
    #         box_hm_prob_size = 2 * level + 1
    #         col = np.reshape(np.array([np.arange(box_hm_prob_size)] * box_hm_prob_size),
    #                          (box_hm_prob_size, box_hm_prob_size))
    #         row = np.transpose(col)
    #         center = level
    #         prob = np.exp(-1 * (np.square(row - center) +
    #                             np.square(col - center)) / self.box_hm_sigma ** 2)
    #         prob_all.append(prob)
    #         return prob_all

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
        images = processing.image_normalization(example)  # 归一化图像
        labels_det = self.batch_genebbox(l_det, l_cg)
        labels_kp = gene_hm.batch_genehm_for_coco(self.coco_batch_size, l_kp, l_det, l_np, l_cg)  # heatmap label
        # for i in range(self.hg_batch_size):
        #     for j in range(5):
        #         print(np.max(labels[i][j]))
        #         plt.imshow(labels[i][j])
        #     plt.imshow(images[i])
        #     plt.matshow(np.sum(labels[i],axis=0))
        #     plt.show()

        return images, labels_det, labels_kp

    # def get_box_hm_prob(self, y_ind, x_ind, label, box):
    #     lt, rt, tp, dn = y_ind - self.box_hm_level, y_ind + self.box_hm_level + 1, \
    #                      x_ind - self.box_hm_level, x_ind + self.box_hm_level + 1
    #     left, right, top, down = max(lt, 0), min(rt, self.cell_size), \
    #                              max(tp, 0), min(dn, self.cell_size)
    #     lt_mg, rt_mg, tp_mg, dn_mg = left - lt, right - rt, top - tp, down - dn
    #     prob = self.box_hm_prob[lt_mg:self.box_hm_prob_size + rt_mg,
    #            tp_mg:self.box_hm_prob_size + dn_mg]
    #     condition = label[left:right, top:down, 0] > prob
    #     prob = np.where(condition,
    #                     label[left:right, top:down, 0], prob)
    #     hm_box = np.where(condition[:, :, np.newaxis],
    #                       label[left:right, top:down, 1:5],
    #                       box[np.newaxis, np.newaxis, :])
    #     return left, right, top, down, prob, hm_box

    def gn_box_hm_prob(self, xmin, xmax, ymin, ymax, label):
        l, r, t, d = map(lambda x: int(x * self.cell_size / self.image_size), (xmin, xmax, ymin, ymax))
        grid_w, grid_h = r + 1 - l, d + 1 - t
        imag_w, imag_h = xmax + 1 - xmin, ymax + 1 - ymin
        # consider image area
        factor = 15 / np.power(imag_h * imag_w, 11 / 20)
        sigma_w, sigma_h = factor * imag_w / 2, factor * imag_h / 2

        col = np.reshape(np.array([np.arange(grid_w)] * grid_h), (grid_h, grid_w))
        row = np.transpose(np.reshape(np.array([np.arange(grid_h)] * grid_w), (grid_w, grid_h)))
        center_col, center_row = (grid_w - 1) / 2, (grid_h - 1) / 2
        prob = np.exp(-1 * ((np.square(col - center_col)) / sigma_w ** 2
                            + (np.square(row - center_row)) / sigma_h ** 2))

        boxes = np.array([(xmax + xmin) / 2.0, (ymax + ymin) / 2.0, xmax - xmin, ymax - ymin])
        condition = label[t:d + 1, l:r + 1, 0] > prob
        label[t:d + 1, l:r + 1, 0] = np.where(condition, label[t:d + 1, l:r + 1, 0], prob)
        label[t:d + 1, l:r + 1, 1:5] = np.where(condition[:, :, np.newaxis],
                                                label[t:d + 1, l:r + 1, 1:5],
                                                boxes[np.newaxis, np.newaxis, :])

    def batch_genebbox(self, batch_det, batch_cg):

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
                x1 = max(min(obj[0], self.image_size - 1), 0)
                y1 = max(min(obj[1], self.image_size - 1), 0)
                x2 = max(min(obj[2], self.image_size - 1), 0)
                y2 = max(min(obj[3], self.image_size - 1), 0)
                boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
                x_ind = int(boxes[0] * self.cell_size / self.image_size)
                y_ind = int(boxes[1] * self.cell_size / self.image_size)

                if self.box_hm:
                    self.gn_box_hm_prob(x1, x2, y1, y2, label)
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
