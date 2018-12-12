import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import hg_yolo.config as cfg
import random
import tensorflow as tf
from dataset import gene_hm
from dataset import processing
from dataset.new_prepro import data_enhance, read_coco_tf
import matplotlib.pyplot as plt


class Coco(object):
    def __init__(self, phase):
        self.sess = None
        #self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        #self.data_path = os.path.join(self.devkil_path, 'VOC2012')
        #self.cache_path = cfg.CACHE_PATH
        self.coco_batch_size = cfg.COCO_BATCH_SIZE
        #self.hg_batch_size = cfg.HOURGLASS_BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.classes = cfg.COCO_CLASSES
        self.num_class = len(self.classes)
        #self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        #self.flipped = cfg.FLIPPED
        self.phase = phase
        #self.rebuild = rebuild
        #self.yolo_cursor = 0
        #self.sum_yolo_cursor = 0
        #self.hg_cursor = 0
        #self.yolo_epoch = 1
        #self.hg_epoch = 1
        #self.gt_labels = self.prepare()
        self.nPoints = cfg.COCO_NPOINTS
        # self.WIDTH = cfg.WIDTH
        # self.HEIGHT = cfg.HEIGHT
        # self.HM_HEIGHT = cfg.HM_HEIGHT
        # self.HM_WIDTH = cfg.HM_WIDTH
        self.coco_filename = cfg.COCO_FILENAME
        # to keep VOC and hg dataset' iteration numbers same in an epoch
        #self.factor = cfg.EPOCH_SIZE * cfg.BATCH_SIZE / len(self.gt_labels)
        self.coco_epoch_size = cfg.COCO_EPOCH_SIZE
        self.images, self.labels_det, self.labels_kp \
            = read_coco_tf.batch_samples(self.coco_batch_size,
                                         self.coco_filename,
                                         shuffle=True)

    def get(self):
        example, l_det, l_kp = self.sess.run([self.images, self.labels_det, self.labels_kp])
        while np.any(np.isnan(example)) or np.any(np.isnan(l_det)) or np.any(np.isnan(l_kp)):
            print('no images or no label')
            example, l_det, l_kp = self.sess.run([self.images, self.labels_det, self.labels_kp])
        images = processing.image_normalization(example)  # 归一化图像
        # images, labels_det, labels_kp = read_coco_tf.batch_samples(self.coco_batch_size,
        #                                                            self.coco_filename,
        #                                                            shuffle=True)
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
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        #imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        #im = cv2.imread(imname)
        #h_ratio = 1.0 * self.image_size / w
        #w_ratio = 1.0 * self.image_size / h
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label_ch = self.num_class + 5
        if self.num_class == 1:
            label_ch = 5
        labels = np.zeros(
            (self.coco_batch_size, self.cell_size, self.cell_size, label_ch))

        for i in range(self.coco_batch_size):
            l_det = batch[i]
            label = np.zeros((self.cell_size, self.cell_size, label_ch))
        #filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        #tree = ET.parse(filename)
        #objs = tree.findall('object')
            l_det = np.reshape(l_det, (-1, 4))
            for obj in l_det:
                # Make pixel indexes 0-based
                # x1 = max(min(obj[0], self.image_size - 1), 0)
                # x2 = max(min((obj[1] - 1), self.image_size - 1), 0)
                # y1 = max(min((obj[2] - 1), self.image_size - 1), 0)
                # y2 = max(min((obj[3] - 1), self.image_size - 1), 0)
                x1 = max(min(obj[0], self.image_size - 1), 0)
                y1 = max(min(obj[1], self.image_size - 1), 0)
                x2 = max(min(obj[2], self.image_size - 1), 0)
                y2 = max(min(obj[3], self.image_size - 1), 0)
                #print(x1, y1, x2, y2)
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
                if cls_ind != 0:
                    label[y_ind, x_ind, 5 + cls_ind] = 1
            labels[i, :, :, :] = label

        return labels
