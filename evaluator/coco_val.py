import os
import cv2
import numpy as np
import tensorflow as tf
from utils import config as cfg
import json
import math
from evaluator.Eutils.draw_result import draw_result


class COCO_VAL(object):

    def __init__(self):
        self.annotations_size = 0
        self.annotations_file = cfg.COCO_ANNOTATION_FILE
        self.image_file = cfg.COCO_VAL_IMAGE_FILE
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
                print(
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
        num_iscrowd = 0
        gt = 0
        for image in images:
            # if gt > 800:
            #     break
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
                if object_annotations['iscrowd'] == 1:
                    num_iscrowd += 1
                    continue
                (x, y, width, height) = tuple(object_annotations['bbox'])
                if width <= 0 or height <= 0:
                    num_annotations_skipped += 1
                    continue
                if x + width > image_width or y + height > image_height:
                    num_annotations_skipped += 1
                    continue
                w = float(width) / image_width * self.image_size
                h = float(height) / image_height * self.image_size
                x = float(x) / image_width * self.image_size + w / 2
                y = float(y) / image_height * self.image_size + h / 2
                bboxes.append([x, y, w, h])
                gt += 1

            if len(bboxes) == 0:
                num_anno_error += 1
                continue

            # bbox_id: {'id': image_id,
            #           'bbox_det': {'bboxes': [[x, y, w, h],...],
            #                        'det': [False,...]}}
            bboxes_det = dict(bboxes=np.array(bboxes), det=[False] * len(bboxes))
            bbox_id = dict(id=image_id, bbox_det=bboxes_det)
            bbox_mat.append(bbox_id)
            self.annotations_size += 1

            # read picture
            filename = image['file_name']
            full_path = os.path.join(self.image_file, filename)
            img = cv2.imread(full_path)
            inputs = cv2.resize(img, (self.image_size, self.image_size))
            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)

            # inputs = inputs.astype(np.float32)
            # myimg = (inputs / 255.0)
            # draw_result(myimg, np.array(bboxes), (0, 0, 255))
            # # draw_result(image, boxes_ft_prob, (255, 0, 0))
            # cv2.imshow('Image', myimg)
            # cv2.waitKey(0)

            inputs = (inputs / 255.0) * 2.0 - 1.0
            inputs = np.reshape(inputs, (self.image_size, self.image_size, 3))
            image_mat.append(inputs)

        print('{} images are missing annotations.'.format(missing_annotation_count))
        print('{} annotations has no bbox.'.format(num_anno_error))
        print('{} annotations are crowd.'.format(num_iscrowd))

        return image_mat, bbox_mat, gt
