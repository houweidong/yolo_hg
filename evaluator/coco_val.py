import os
import cv2
import numpy as np
import tensorflow as tf
from utils import config as cfg
import json


class COCO_VAL(object):

    def __init__(self):
        self.annotations_size = 0
        self.annotations_file = cfg.COCO_ANNOTATION_FILE
        self.image_file = cfg.COCO_VAL_IMAGE_FILE
        self.batch_size = cfg.COCO_BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cursor = 0
        self.image_mat, self.bbox_mat = self.prepare_data()

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
            image_batch = None
            bbox_batch = None
        return image_batch, bbox_batch

    def prepare_data(self):
        # read annotation from coco file
        with tf.gfile.GFile(self.annotations_file, 'r') as fid:
            groundtruth_data = json.load(fid)  # json file
            images = groundtruth_data['images']

            annotations_index = {}
            if 'annotations' in groundtruth_data:
                tf.logging.info(
                    'Found {:<5} groundtruth annotations. Building annotations index.'
                    .format(len(groundtruth_data['annotations'])))
                for annotation in groundtruth_data['annotations']:
                    image_id = annotation['image_id']
                    if image_id not in annotations_index:
                        annotations_index[image_id] = []
                    annotations_index[image_id].append(annotation)

        # parse all images and det labels
        image_mat = []
        bbox_mat = []
        missing_annotation_count = 0
        num_annotations_skipped = 0
        num_anno_error = 0  # annotation has no bbox
        for image in images:
            image_id = image['id']
            if image_id not in annotations_index:
                missing_annotation_count += 1
                continue

            # read det label
            annotations_list = annotations_index[image_id]
            image_height = image['height']
            image_width = image['width']
            bboxes = []
            for object_annotations in annotations_list:
                (x, y, width, height) = tuple(object_annotations['bbox'])
                if width <= 0 or height <= 0:
                    num_annotations_skipped += 1
                    continue
                if x + width > image_width or y + height > image_height:
                    num_annotations_skipped += 1
                    continue
                bboxes.append(object_annotations['bbox'])

            if len(bboxes) == 0:
                num_anno_error += 1
                continue
            bbox_mat.append(bboxes)
            self.annotations_size += 1

            # read picture
            filename = image['file_name']
            full_path = os.path.join(self.image_file, filename)
            img = cv2.imread(full_path)
            inputs = cv2.resize(img, (self.image_size, self.image_size))
            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
            inputs = (inputs / 255.0) * 2.0 - 1.0
            inputs = np.reshape(inputs, (self.image_size, self.image_size, 3))
            image_mat.append(inputs)

        tf.logging.info('%d images are missing annotations.',
                        missing_annotation_count)
        tf.logging.info('%d annotations has no bbox.',
                        num_anno_error)

        return image_mat, bbox_mat
