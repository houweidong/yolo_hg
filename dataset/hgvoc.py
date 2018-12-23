import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
from utils import config as cfg
import random
import tensorflow as tf
from dataset.Dutils import gene_hm
from dataset.Dutils import processing


class hg_voc(object):
    def __init__(self, phase, option, rebuild=False):
        self.sess = None
        self.option = option
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        self.data_path = os.path.join(self.devkil_path, 'VOC2012')
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.hg_batch_size = cfg.HOURGLASS_BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.flipped = cfg.FLIPPED
        self.phase = phase
        self.rebuild = rebuild
        self.yolo_cursor = 0
        self.sum_yolo_cursor = 0
        self.hg_cursor = 0
        self.epoch = 1
        self.yolo_epoch = 1
        self.hg_epoch = 1
        self.gt_labels = self.prepare()
        self.nPoints = cfg.NPOINTS
        self.WIDTH = cfg.WIDTH
        self.HEIGHT = cfg.HEIGHT
        self.HM_HEIGHT = cfg.HM_HEIGHT
        self.HM_WIDTH = cfg.HM_WIDTH
        self.filename = cfg.FILENAME
        # to keep VOC and hg dataset' iteration numbers same in an epoch
        self.factor = cfg.EPOCH_SIZE * cfg.BATCH_SIZE / len(self.gt_labels)
        self.epoch_size = cfg.EPOCH_SIZE
        self.images_hg, self.labels_hg = self.batch_samples(self.hg_batch_size,
                                                            self.filename,
                                                            self.nPoints,
                                                            True)

    def all_get(self):
        if self.option == 1 and self.yolo_epoch > self.epoch and self.hg_epoch > self.epoch:
            self.epoch += 1
        # bound = self.epoch_size / (len(self.gt_labels) / 50 * self.factor + self.epoch_size)
        bound = 0.5
        if self.option == 1:
            seed = random.random()
        elif self.option == 2:
            seed = 2
        elif self.option == 3:
            seed = -1
        else:
            raise Exception("self.option is not in [1, 2, 3]!")

        if self.option == 3 \
                or (self.option == 1 and self.yolo_epoch > self.epoch)\
                or (self.option == 1 and self.hg_epoch == self.epoch and seed < bound):
            images_labels = self.hg_get()
            while not images_labels:
                images_labels = self.hg_get()
            images, labels = images_labels
            sign = "hourglass"
        else:
            images_labels = self.get()
            while not images_labels:
                images_labels = self.get()
            images, labels = images_labels
            sign = "VOC "
        return images, labels, sign

    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 25))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.yolo_cursor]['imname']
            flipped = self.gt_labels[self.yolo_cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.yolo_cursor]['label']
            count += 1
            self.yolo_cursor = (self.yolo_cursor + 1) % len(self.gt_labels)
            if self.yolo_cursor == 0:
                np.random.shuffle(self.gt_labels)
            if self.option == len(self.gt_labels):
                self.sum_yolo_cursor += 1
                # if self.yolo_cursor >= len(self.gt_labels):
                #     np.random.shuffle(self.gt_labels)
                #     self.yolo_cursor = 0
                if self.sum_yolo_cursor >= len(self.gt_labels) * self.factor:
                    self.yolo_epoch += 1
                    self.sum_yolo_cursor = 0
            else:  # self.option is 2
                if self.yolo_cursor == 0:
                    self.epoch += 1
        if np.any(np.isnan(images)):
            print('no images in yolo')
            return None
        if np.any(np.isnan(labels)):
            print('no label in yolo')
            return None
        return images, labels

    def hg_get(self):
        example, l = self.sess.run([self.images_hg, self.labels_hg])
        if np.any(np.isnan(example)):
            print('no images in hourglass')
            return None
        if np.any(np.isnan(l)):
            print('no label in houtglass')
            return None
        images = processing.image_normalization(example)  # 归一化图像
        labels = gene_hm.batch_genehm(self.hg_batch_size, l)  # heatmap label

        # for i in range(self.hg_batch_size):
        #     for j in range(5):
        #         print(np.max(labels[i][j]))
        #         plt.imshow(labels[i][j])
        #     plt.imshow(images[i])
        #     plt.matshow(np.sum(labels[i],axis=0))
        #     plt.show()

        # if self.option == 1:
        self.hg_cursor += 1
        if self.hg_cursor >= self.epoch_size:
            self.hg_cursor = 0
            self.hg_epoch += 1
            if self.option == 3:
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] =\
                    gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 -\
                                gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'test.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 25))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)

    # ***********************
    # dataset of reading houglass dataset
    def _read_single_sample(self, samples_dir, nPoints):

        filename_quene = tf.train.string_input_producer([samples_dir])
        reader = tf.TFRecordReader()
        _, serialize_example = reader.read(filename_quene)
        features = tf.parse_single_example(
            serialize_example,
            features={
                'label': tf.FixedLenFeature([nPoints * 2], tf.float32),
                'image': tf.FixedLenFeature([], tf.string)
            }
        )

        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, [self.HEIGHT, self.WIDTH, 3])  # ！reshape 先列后行
        label = tf.cast(features['label'], tf.float32)

        return image, label

    def resize_img_label(self, image, label, width, height):
        new_img = tf.image.resize_images(image, [256, 256], method=1)
        x = tf.reshape(label[:, 0] * 256. / tf.cast(width, tf.float32), (-1, 1))
        y = tf.reshape(label[:, 1] * 256. / tf.cast(height, tf.float32), (-1, 1))
        re_label = tf.concat([x, y], axis=1)
        return new_img, re_label

    def batch_samples(self, batch_size, filename, nPoints, shuffle=True):
        """
        filename:tfrecord文件名
        """

        image, label = self._read_single_sample(filename, nPoints)
        # print(image.shape)
        # label=tf.reshape(label,[-1,2])

        image, label, re_width, re_height = self.do_enhance(image, label, 512, 512)
        image, label = self.resize_img_label(image, label, re_width, re_height)
        # image,label=resize_img_label(image,label,512,512)

        # label = gene_hm.resize_label(label)#将label放缩到64*64
        # label=gene_hm.tf_generate_hm(HM_HEIGHT, HM_WIDTH ,label, 64)
        if shuffle:
            image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size,
                                                              min_after_dequeue=batch_size * 5, num_threads=2,
                                                              capacity=batch_size * 300)
        else:
            image_batch, label_batch = tf.train.batch([image, label], batch_size, num_threads=2)

        return image_batch, label_batch

    def _gen_full_boundingBox(self, label, width, height):

        re_width = tf.cast(width, tf.int32)
        re_height = tf.cast(height, tf.int32)
        re_label = tf.reshape(label, (-1, 2))
        l_min = tf.reduce_min(re_label, axis=0)
        l_max = tf.reduce_max(re_label, axis=0)
        left_margin = tf.cast(tf.floor(l_min[0]), tf.int32)
        top_margin = tf.cast(tf.floor(l_min[1]), tf.int32)
        right_margin = tf.cast(tf.floor(l_max[0]), tf.int32)
        bottom_margin = tf.cast(tf.floor(l_max[1]), tf.int32)

        left0 = tf.random_uniform([1], 0, left_margin + 1, dtype=tf.int32)
        left = tf.random_uniform([1], 0, left0[0] + 1, dtype=tf.int32)
        # left=tf.random_uniform([1],0,left1[0]+1,dtype=tf.int32)

        top0 = tf.random_uniform([1], 0, top_margin + 1, dtype=tf.int32)
        top = tf.random_uniform([1], 0, top0[0] + 1, dtype=tf.int32)
        # top=tf.random_uniform([1],0,top1[0]+1,dtype=tf.int32)

        right0 = tf.random_uniform([1], right_margin, re_width, dtype=tf.int32)
        right = tf.random_uniform([1], right0[0], re_width, dtype=tf.int32)
        # right=tf.random_uniform([1],right1[0],re_width,dtype=tf.int32)

        bottom0 = tf.random_uniform([1], bottom_margin, re_height, dtype=tf.int32)
        bottom = tf.random_uniform([1], bottom0[0], re_height, dtype=tf.int32)
        # bottom=tf.random_uniform([1],bottom1[0],re_height,dtype=tf.int32)
        new_width = right - left
        new_height = bottom - top

        return top[0], left[0], new_height[0], new_width[0]

    def _relabel_ac_bbox(self, label, bbox):
        re_label = tf.reshape(label, (-1, 2))

        top = tf.cast(bbox[0], tf.float32)
        left = tf.cast(bbox[1], tf.float32)
        x = tf.reshape(re_label[:, 0] - left, (-1, 1))
        y = tf.reshape(re_label[:, 1] - top, (-1, 1))
        result = tf.concat([x, y], axis=1)
        return result

    def random_crop_img(self, img, label, width, height):
        bbox = self._gen_full_boundingBox(label, width, height)
        crop_img = tf.image.crop_to_bounding_box(img, bbox[0], bbox[1], bbox[2], bbox[3])
        re_label = self._relabel_ac_bbox(label, bbox)
        return crop_img, re_label, bbox[2], bbox[3]

    def adjust_image(self, color_ordering, image, fast=False):
        if fast:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)

            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)

            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)

            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        return image

    def do_enhance(self, image, label, width, height):
        """

        :param image:
        :param label:
        :param width:
        :param height:
        :return: label shape is (nPoints,2)
        """
        crop_image, re_label, re_height, re_width = self.random_crop_img(image, label, width, height)
        adj_image = self.adjust_image(1, crop_image)
        return adj_image, re_label, re_width, re_height
