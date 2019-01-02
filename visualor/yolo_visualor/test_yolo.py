import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from utils import config as cfg
from model.hourglass_yolo_net import HOURGLASSYOLONet
from utils.timer import Timer
import collections

class Detector(object):

    def __init__(self, net, weight_file, values):
        self.fc = values['BOX_FOCAL_LOSS']
        self.net = net
        self.weights_file = weight_file
        # self.classes = cfg.COCO_CLASSES
        # self.num_class = len(self.classes)
        # self.image_size = cfg.IMAGE_SIZE
        # self.cell_size = cfg.CELL_SIZE
        # self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD_NMS
        # self.boundary1 = self.cell_size * self.cell_size * self.num_class
        # self.boundary2 = self.boundary1 +\
        #     self.cell_size * self.cell_size * self.boxes_per_cell
        # self.boundary1 = self.cell_size * self.cell_size * self.num_class if self.num_class != 1 else 0
        # self.boundary2 = self.boundary1 +\
        #     self.cell_size * self.cell_size * self.boxes_per_cell
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        for i in range(len(result)):
            # print(result[i][0])
            # if result[i][0] != 'person':
            #     continue
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.net.image_size, self.net.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.net.image_size, self.net.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.net.image_size)
            result[i][2] *= (1.0 * img_h / self.net.image_size)
            result[i][3] *= (1.0 * img_w / self.net.image_size)
            result[i][4] *= (1.0 * img_h / self.net.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.yolo_logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.net.cell_size, self.net.cell_size,
                          self.net.boxes_per_cell, self.net.num_class))
        class_probs = output[:, :, :self.net.boundary1] if self.net.num_class != 1 \
            else np.ones((self.net.cell_size, self.net.cell_size, 1))
        # if self.num_class != 1:
        #     class_probs = np.reshape(
        #         output[:, :, :self.net.boundary1],
        #         (self.cell_size, self.cell_size, self.num_class))
        # else:
        #     class_probs = np.ones((self.cell_size, self.cell_size, self.num_class))
        scales = output[:, :, self.net.boundary1:self.net.boundary2]
        if self.fc:
            scales = 1 / (1 + np.exp(-scales))
        boxes = np.reshape(
            output[:, :, self.net.boundary2:],
            (self.net.cell_size, self.net.cell_size, self.net.boxes_per_cell, 4))
        # scales = np.reshape(
        #     output[self.boundary1:self.boundary2],
        #     (self.cell_size, self.cell_size, self.boxes_per_cell))
        # boxes = np.reshape(
        #     output[self.boundary2:],
        #     (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.array(
            [np.arange(self.net.cell_size)] * self.net.cell_size * self.net.boxes_per_cell)
        offset = np.transpose(
            np.reshape(
                offset,
                [self.net.boxes_per_cell, self.net.cell_size, self.net.cell_size]),
            (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.net.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.net.image_size

        for i in range(self.net.boxes_per_cell):
            for j in range(self.net.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)

        # [[bbox1] [bbox2] ...]
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        # [0.7, 0.5, 0.91, ...]
        probs_filtered = probs[filter_mat_probs]
        # [2, 3, 0, 20, ...]
        classes_num_filtered = np.argmax(
            probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        # [2, 1, 0, ...]
        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        # NMS
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        # select bbox whose probs is not 0
        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [cfg.COCO_CLASSES[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        self.draw_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)

    def images_detector(self, imspth, wait=0):
        img_names = os.listdir(imspth)
        detect_timer = Timer()
        for name in img_names:
            img_path = os.path.join(imspth, name)
            try:
                image = cv2.imread(img_path)
            except Exception as info:
                print(info)
                continue

            detect_timer.tic()
            result = self.detect(image)
            detect_timer.toc()
            # if len(result) == 0:
            #     continue

            self.draw_result(image, result)
            cv2.imshow('Image', image)

            cv2.waitKey(wait)
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))
        # cv2.waitKey(wait)


def get_config(config_path):
    config = os.path.join(config_path, 'config.txt')
    values = collections.OrderedDict()
    keys = ['ADD_YOLO_POSITION', 'LOSS_FACTOR', 'LEARNING_RATE', 'OBJECT_SCALE',
            'NOOBJECT_SCALE', 'COORD_SCALE', 'BOX_FOCAL_LOSS']
    values = values.fromkeys(keys)
    for line in open(config):
        name, value = line.split(': ')[0], line.split(': ')[1]
        if name in keys:
            values[name] = value.strip()
    cfg.ADD_YOLO_POSITION = values['ADD_YOLO_POSITION']
    if 'fc' in config_path:
        values['BOX_FOCAL_LOSS'] = True
    else:
        values['BOX_FOCAL_LOSS'] = False
    strings = ''
    for i, value in values.items():
        strings += '{}:{}  '.format(i, value)
    return values, strings


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--position', default="tail_conv", type=str,
    #                     choices=["tail", "tail_tsp", "tail_conv", "tail_tsp_self",
    #                              "tail_conv_deep", "tail_conv_deep_fc"])
    # parser.add_argument('--csize', default=64, type=int)
    # parser.add_argument('-fc', '--focal_loss', action='store_true', help='use focal loss')
    parser.add_argument('--weights', default="hg_yolo-240000", type=str)
    parser.add_argument('--weight_dir', default='../../log/10_1_10_conv_fc/', type=str)
    # parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    values, strings = get_config(args.weight_dir)
    # cfg.CELL_SIZE = args.csize
    yolo = HOURGLASSYOLONet('visual')
    detector = Detector(yolo, os.path.join(args.weight_dir, args.weights), values)

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from image file
    ims_pth = "/root/dataset/val2017"
    # ims_pth = "../pictures"
    # ims_pth = "../pictures1/"
    imname = 'pictures/2.jpg'
    detector.images_detector(ims_pth)


if __name__ == '__main__':
    main()
