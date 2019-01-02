import os
import cv2
import numpy as np
from utils import config as cfg
import math
import xml.etree.ElementTree as ET
from evaluator.Eutils.draw_result import draw_result


class PASCAL_VAL(object):

    def __init__(self):
        self.classes = cfg.PASCAL_CLASSES
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.annotations_size = 0
        self.data_path = cfg.PASCAL_DATA
        self.batch_size = cfg.COCO_BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cursor = 0
        self.image_mat, self.bbox_mat, self.gt = self.prepare_data()
        self.num_batch = math.ceil(self.annotations_size / self.batch_size)

    def get_batch(self):
        if self.cursor <= self.annotations_size - self.batch_size:
            image_batch = self.image_mat[self.cursor:self.cursor+self.batch_size]
            bbox_batch = self.bbox_mat[self.cursor:self.cursor+self.batch_size]
            self.cursor += self.batch_size
        elif self.cursor < self.annotations_size:
            image_batch = self.image_mat[self.cursor:]
            bbox_batch = self.bbox_mat[self.cursor:]
            self.cursor = self.annotations_size
        else:
            # start another epoch
            self.cursor = 0
            image_batch, bbox_batch = self.get_batch()
        return image_batch, bbox_batch

    def prepare_data(self):

        print('Processing gt_labels from: ' + self.data_path)
        txtname = os.path.join(
            self.data_path, 'ImageSets', 'Main', 'val.txt')
        with open(txtname, 'r') as f:
            image_index = [x.strip() for x in f.readlines()]
        # parse all images and det labels
        image_mat = []
        bbox_mat = []
        num_anno_no_person = 0
        num_gt = 0
        for index in image_index:
            # if num_gt > 80:
            #     break
            # read det label
            bboxes, gt = self.load_pascal_annotation(index)
            num_gt += gt
            if len(bboxes) == 0:
                num_anno_no_person += 1

            # bbox_id: {'id': image_id,
            #           'bbox_det': {'bboxes': [[x, y, w, h],...],
            #                        'det': [False,...]}}
            bboxes_det = dict(bboxes=np.array(bboxes), det=[False] * len(bboxes))
            bbox_id = dict(id=index, bbox_det=bboxes_det)
            bbox_mat.append(bbox_id)
            self.annotations_size += 1

            # read picture
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            image = self.image_read(imname)
            image_mat.append(image)
        print('{} annotations has no bbox.'.format(num_anno_no_person))
        return image_mat, bbox_mat, num_gt

    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        gt = 0
        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        bboxes = []
        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            # person -> 14
            if cls_ind != 14:
                continue
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            xywh = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            bboxes.append(xywh)
            gt += 1
        return bboxes, gt

    def image_read(self, imname):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        image = np.reshape(image, (self.image_size, self.image_size, 3))
        return image

