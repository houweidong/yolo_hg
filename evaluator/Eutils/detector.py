import numpy as np
import tensorflow as tf
from evaluator.Eutils.nms import py_cpu_nms


class Detector(object):

    def __init__(self, net, weight_file, values):
        self.focal_loss = values['BOX_FOCAL_LOSS']
        self.net = net
        self.weights_file = weight_file
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # tf.reset_default_graph()
        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def detect_batch(self, batch):
        """

        :param batch[batch_size, ]  (image[width, height, 3] numpy)
        :return: results[batch_size, ]  sorted according prob(bboxes[n, 4], prob[n, ], class[n, ]  numpy)
        """
        net_output = self.sess.run(self.net.yolo_logits,
                                   feed_dict={self.net.images: np.array(batch)})
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))
        return results

    def interpret_output(self, output):
        """

        :param output: yolo head of yolo_hg net of one picture
        :return: bboxes[n, 4], prob[n, ], class[n, ]  type: numpy
        """
        # probs = np.zeros((self.net.cell_size, self.net.cell_size,
        #                   self.net.boxes_per_cell, self.net.num_class))
        class_probs = output[:, :, :self.net.boundary1] if self.net.num_class != 1 \
            else np.ones((self.net.cell_size, self.net.cell_size, 1))

        scales = output[:, :, self.net.boundary1:self.net.boundary2]
        if self.focal_loss:
            # if inx >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
            #     return 1.0 / (1 + exp(-inx))
            # else:
            #     return exp(inx) / (1 + exp(inx))
            # scales = 1 / (1 + np.exp(-scales))
            scales = np.clip(scales, -50, 50)
            scales = 1 / (1 + np.exp(-scales))

        boxes = np.reshape(
            output[:, :, self.net.boundary2:],
            (self.net.cell_size, self.net.cell_size, self.net.boxes_per_cell, 4))

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

        scales = scales[..., np.newaxis]
        class_probs = class_probs[:, :, np.newaxis, :]
        probs = np.multiply(class_probs, scales)
        # for i in range(self.net.boxes_per_cell):
        #     for j in range(self.net.num_class):
        #         probs[:, :, i, j] = np.multiply(
        #             class_probs[:, :, j], scales[:, :, i])

        probs = np.clip(probs, 0, 1)
        filter_mat_probs = np.array(probs >= 0.0, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)

        # [[bbox1] [bbox2] ...]
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        # [0.7, 0.5, 0.91, ...]
        probs_filtered = probs[filter_mat_probs]
        # [2, 3, 0, 20, ...]
        classes_num_filtered = np.argmax(
            probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        # [2, 1, 0, ...]
        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        # NMS
        keep = py_cpu_nms(boxes_filtered)
        # select bbox should keep
        boxes_filtered = boxes_filtered[keep]
        probs_filtered = probs_filtered[keep]
        classes_num_filtered = classes_num_filtered[keep]

        # select bboxes whose category id are 1(person)
        mask_p = classes_num_filtered.astype(np.bool) if self.net.num_class != 1 \
            else [True] * len(classes_num_filtered)
        boxes_person = boxes_filtered[mask_p]
        probs_person = probs_filtered[mask_p]
        return boxes_person, probs_person
