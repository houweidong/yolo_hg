import numpy as np
from utils import config as cfg
from dataset.Dutils import read_coco_tf, gene_hm, processing
# import matplotlib.pyplot as plt


class Coco(object):
    def __init__(self):
        self.sess = None
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
        self.train_im_batch, self.train_labels_det_batch, self.train_labels_kp_batch \
            = read_coco_tf.batch_samples(self.coco_batch_size,
                                         self.coco_train_fn,
                                         shuffle=True)
        self.val_im_batch, self.val_labels_det_batch, self.val_labels_kp_batch \
            = read_coco_tf.batch_samples(self.coco_batch_size,
                                         self.coco_val_fn,
                                         shuffle=False)

    def get(self, phase):
        # while np.any(np.isnan(example)) or np.any(np.isnan(l_det)) or np.any(np.isnan(l_kp)):
        #     print('no images or no label')
        #     example, l_det, l_kp = self.sess.run([self.train_im_batch,
        #                                           self.train_labels_det_batch,
        #                                           self.train_labels_kp_batch])
        if phase == "train":
            example, l_det, l_kp = self.sess.run([self.train_im_batch,
                                                  self.train_labels_det_batch,
                                                  self.train_labels_kp_batch])
        else:
            example, l_det, l_kp = self.sess.run([self.val_im_batch,
                                                  self.val_labels_det_batch,
                                                  self.val_labels_kp_batch])
        images = processing.image_normalization(example)  # 归一化图像
        labels_det = self.batch_genebbox(l_det)
        labels_kp = gene_hm.batch_genehm_for_coco(self.coco_batch_size, l_kp, self.nPoints)  # heatmap label
        # for i in range(self.hg_batch_size):
        #     for j in range(5):
        #         print(np.max(labels[i][j]))
        #         plt.imshow(labels[i][j])
        #     plt.imshow(images[i])
        #     plt.matshow(np.sum(labels[i],axis=0))
        #     plt.show()

        return images, labels_det, labels_kp

    def batch_genebbox(self, batch):

        label_ch = self.num_class + 5
        rs_ch = 5
        if self.num_class == 1:
            label_ch = 5
            rs_ch = 4
        labels = np.zeros(
            (self.coco_batch_size, self.cell_size, self.cell_size, label_ch))

        for i in range(self.coco_batch_size):
            l_det = batch[i]
            l_det = np.reshape(l_det, (-1, rs_ch))
            label = np.zeros((self.cell_size, self.cell_size, label_ch))
            for obj in l_det:
                if np.array_equal(obj, [256, 0, 256, 0]) or np.array_equal(obj, [0, 0, 0, 0]):
                    continue
                x1 = max(min(obj[0], self.image_size - 1), 0)
                y1 = max(min(obj[1], self.image_size - 1), 0)
                x2 = max(min(obj[2], self.image_size - 1), 0)
                y2 = max(min(obj[3], self.image_size - 1), 0)
                cls_ind = 0
                if len(obj) == 5:
                    cls_ind = obj[4]
                boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
                x_ind = int(boxes[0] * self.cell_size / self.image_size)
                y_ind = int(boxes[1] * self.cell_size / self.image_size)
                if label[y_ind, x_ind, 0] == 1:
                    continue
                label[y_ind, x_ind, 0] = 1
                label[y_ind, x_ind, 1:5] = boxes

                # multi classification
                if self.num_class != 1:
                    label[y_ind, x_ind, 5 + cls_ind] = 1
            labels[i, :, :, :] = label

        return labels
