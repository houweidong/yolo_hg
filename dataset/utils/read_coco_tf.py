import tensorflow as tf
from dataset.utils import data_enhance
from utils import config as cfg
import os

"""加载一个batchsize的image"""
WIDTH = 256
HEIGHT = 256
HM_HEIGHT = 64
HM_WIDTH = 64
MAX = cfg.COCO_MAX_PERSON_PER_PIC


def _read_single_sample(samples_dir):
    ab_dir = []
    for tr_path in os.listdir(samples_dir):
        ab_dir.append(os.path.join(samples_dir, tr_path))
    filename_quene = tf.train.string_input_producer(ab_dir)
    reader = tf.TFRecordReader()
    _, serialize_example = reader.read(filename_quene)
    features = tf.parse_single_example(
        serialize_example,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/object/bboxes': tf.FixedLenFeature([MAX * 4], tf.float32),
            'image/object/keypoints': tf.FixedLenFeature([MAX * cfg.COCO_NPOINTS * 3], tf.int64)

        }
    )
    width = tf.cast(features['image/width'], tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.reshape(image, [height, width, 3])  # ！reshape 先列后行
    xxyy = tf.cast(features['image/object/bboxes'], tf.float32)
    keypoints = tf.cast(features['image/object/keypoints'], tf.int32)

    return image, keypoints, width, height, xxyy
    # print(img.shape)
    # print(label)


def resize_img_label(image, label, width, height, gt_bbox):
    new_img = tf.image.resize_images(image, [256, 256], method=1)
    x = tf.reshape(label[:, 0] * 256. / tf.cast(width, tf.float32), (-1, 1))
    y = tf.reshape(label[:, 1] * 256. / tf.cast(height, tf.float32), (-1, 1))
    re_label = tf.concat([x, y], axis=1)
    gt_bbox = tf.cast(gt_bbox, tf.float32)
    b_x = tf.reshape(gt_bbox[:, 0] * 256. / tf.cast(width, tf.float32), (-1, 1))
    b_y = tf.reshape(gt_bbox[:, 1] * 256. / tf.cast(height, tf.float32), (-1, 1))
    re_bbox = tf.concat([b_x, b_y], axis=1)

    return new_img, re_label, re_bbox


def batch_samples(batch_size, filename, shuffle=False):
    """
    filename:tfrecord文件名
    """

    image, label, width, height, bbox = _read_single_sample(filename)

    label = tf.cast(tf.reshape(label, [-1, 3]), tf.float32)
    bbox = tf.cast(tf.reshape(bbox, [-1, 2]), tf.int64)
    [image, label, bbox], new_width, new_height = data_enhance.do_enhance(image, label, width, height, True, bbox)
    # image, label, bbox = resize_img_label(image, label, width, height, bbox)
    image, label, bbox = resize_img_label(image, label, new_width, new_height, bbox)
    # label.set_shape([cfg.COCO_NPOINTS * MAX, 2])
    label = tf.cast(tf.reshape(label, [cfg.COCO_NPOINTS * MAX, 2]), tf.float32)
    bbox = tf.cast(tf.reshape(bbox, (-1, 2)), tf.int64)

    if shuffle:
        b_image, b_label, b_bbox = tf.train.shuffle_batch([image, label, bbox], batch_size,
                                                          min_after_dequeue=batch_size * 5, num_threads=2,
                                                          capacity=batch_size * 300)
    else:
        b_image, b_label, b_bbox = tf.train.batch([image, label, bbox], batch_size, num_threads=2)

    return b_image, b_bbox, b_label

# # # """测试加载图像"""
# import matplotlib.pyplot as plt
# import numpy as np

# with tf.Session() as sess:  # 开始一个会话
#     init_op = tf.global_variables_initializer()
#     tf.local_variables_initializer().run()
#     sess.run(init_op)
#     # tf.local_variables_initializer().run()
#
#     b_image, b_bbox, b_label = batch_samples(1,
#                                              '/root/dataset/tfrecord/val/coco_val.record-00000-of-00003',
#                                              False)
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(7900):
#         try:
#             r_image, r_label, r_bbox = sess.run([b_image, b_label, b_bbox])  # 在会话中取出image和label
#         except tf.errors.OutOfRangeError as info:
#             print('info', info)
#             exit()
#         else:
#             #
#             # print(r_image.shape)
#             # print(r_label.shape)
#             #
#             print(i)
#
#             for j in range(1):
#
#                 x = r_label[j][:, 0]
#                 y = r_label[j][:, 1]
#                 bbox = np.reshape(r_bbox[j], (-1, 2))
#                 print(r_label[j])
#
#                 print(bbox)
#                 plt.imshow(r_image[j], cmap='Greys_r')
#                 plt.plot(x, y, 'r+')
#
#                 leng = r_bbox.shape[1]
#                 for ii in range(leng // 2):
#                     rect = plt.Rectangle((r_bbox[j][ii * 2][0], r_bbox[j][ii * 2][1]),
#                                          r_bbox[j][ii * 2 + 1][0] - r_bbox[j][ii * 2][0],
#                                          r_bbox[j][ii * 2 + 1][1] - r_bbox[j][ii * 2][1],
#                                          linewidth=1,
#                                          edgecolor='r',
#                                          facecolor='none')
#                 # rect1 = plt.Rectangle((r_bbox[j][2][0], r_bbox[j][2][1]), r_bbox[j][3][0] - r_bbox[j][2][0],
#                 #                       r_bbox[j][3][1] - r_bbox[j][2][1], linewidth=1, edgecolor='r', facecolor='none')
#
#                     plt.gca().add_patch(rect)
#                 # plt.gca().add_patch(rect1)
#                 plt.show()
#     coord.request_stop()
#     coord.join(threads)