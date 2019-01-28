import tensorflow as tf
# from dataset.Dutils import data_enhance,gene_hm
from utils import config as cfg
import os

"""加载一个batchsize的image"""
MAX_OBJECT = cfg.COCO_MAX_OBJECT_PER_PIC


def _read_single_sample_all_categories(samples_dir):
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
            'image/object/bboxes': tf.FixedLenFeature([MAX_OBJECT * 4], tf.float32),
            'image/object/keypoints': tf.FixedLenFeature([MAX_OBJECT * cfg.COCO_NPOINTS * 3], tf.int64),
            'image/object/category_id': tf.FixedLenFeature([MAX_OBJECT], tf.int64),
            'image/object/num_keypoints': tf.FixedLenFeature([MAX_OBJECT], tf.int64)
        }
    )
    width = tf.cast(features['image/width'], tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.reshape(image, [height, width, 3])  # ！reshape 先列后行
    xxyy = tf.cast(features['image/object/bboxes'], tf.float32)
    keypoints = tf.cast(features['image/object/keypoints'], tf.int32)
    category_id = tf.cast(features['image/object/category_id'], tf.int32)
    num_keypoints = tf.cast(features['image/object/num_keypoints'], tf.int32)
    return image, keypoints, width, height, xxyy, category_id, num_keypoints
    # return image, (width, height), (keypoints, num_keypoints), (xxyy, category_id)
    # print(img.shape)
    # print(label)


def resize_img_label(image, label, width, height, gt_bbox):
    label = tf.cast(tf.reshape(label, [-1, 3]), tf.float32)
    gt_bbox = tf.cast(tf.reshape(gt_bbox, [-1, 2]), tf.float32)
    new_img = tf.image.resize_images(image, [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE], method=1)

    x = tf.reshape(label[:, 0], (-1, 1))
    y = tf.reshape(label[:, 1], (-1, 1))
    re_label_origin = tf.concat([x, y], axis=1)
    # translate the key points coord to continuous form
    pad = tf.constant(0, dtype=tf.float32, shape=[cfg.COCO_NPOINTS * MAX_OBJECT, 2])
    mask_int = tf.cast(tf.equal(re_label_origin, pad), tf.int32)
    mask_int_1d = tf.cast(tf.reduce_sum(mask_int, 1), tf.bool)
    re_label_origin = tf.where(mask_int_1d, pad, re_label_origin + tf.constant(0.5, dtype=tf.float32))

    x = tf.reshape(re_label_origin[:, 0] * float(cfg.IMAGE_SIZE) / tf.cast(width, tf.float32), (-1, 1))
    y = tf.reshape(re_label_origin[:, 1] * float(cfg.IMAGE_SIZE) / tf.cast(height, tf.float32), (-1, 1))
    re_label = tf.concat([x, y], axis=1)

    b_x = tf.reshape(gt_bbox[:, 0] * float(cfg.IMAGE_SIZE) / tf.cast(width, tf.float32), (-1, 1))
    b_y = tf.reshape(gt_bbox[:, 1] * float(cfg.IMAGE_SIZE) / tf.cast(height, tf.float32), (-1, 1))
    re_bbox = tf.concat([b_x, b_y], axis=1)
    # re_label = tf.reshape(re_label, [cfg.COCO_NPOINTS * MAX_OBJECT, 2])
    # re_bbox = tf.reshape(re_bbox, (-1, 2))
    return new_img, re_label, re_bbox


def batch_samples_all_categories(batch_size, filename, shuffle=False):
    """
    filename:tfrecord文件名
    """

    image, label, width, height, bbox, category_id, num_keypoints = \
        _read_single_sample_all_categories(filename)
    image, label, bbox = resize_img_label(image, label, width, height, bbox)

    if shuffle:
        b_image, b_bbox, category_id, b_label, b_num_kpoints = \
            tf.train.shuffle_batch([image, bbox, category_id, label, num_keypoints],
                                   batch_size,
                                   min_after_dequeue=batch_size * 5, num_threads=2,
                                   capacity=batch_size * 300)
    else:
        b_image, b_bbox, category_id, b_label, b_num_kpoints = \
            tf.train.batch([image, bbox, category_id, label, num_keypoints], batch_size, num_threads=2)

    return b_image, b_bbox, category_id, b_label, b_num_kpoints


# # # """测试加载图像"""
# from dataset.coco import Coco
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
#
# with tf.Session() as sess:  # 开始一个会话
#     coco = Coco()
#     coco.coco_batch_size = 1
#     coco.sess = sess
#     init_op = tf.global_variables_initializer()
#     tf.local_variables_initializer().run()
#     sess.run(init_op)
#     batch_size = 1
#
#     b_image, b_bbox, b_category_id, b_label, b_num_kpoints = \
#         batch_samples_all_categories(batch_size,
#                                      '/root/dataset/tfrecord_only_person/val',
#                                      False)
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(7900):
#         try:
#             # 在会话中取出image和label
#             r_image, r_label, r_bbox, r_num_kpoints, r_category_id = sess.run(
#                 [b_image, b_label, b_bbox, b_num_kpoints, b_category_id])
#             r_bbox = r_bbox * 64 / 256
#
#         except tf.errors.OutOfRangeError as info:
#             print('info', info)
#             exit()
#         else:
#             print(i)
#             # print(r_category_id)
#             hm_result = coco.batch_gene_hm_kp(r_label, r_bbox, r_num_kpoints, r_category_id)
#             for j in range(batch_size):
#                 or_x = r_label[j][:, 0] * 64 / 256
#                 or_y = r_label[j][:, 1] * 64 / 256
#
#                 img = Image.fromarray(r_image[j])
#                 plt.imshow(img, cmap='Greys_r')
#                 plt.matshow(np.sum(hm_result[j], axis=0))
#                 all_x = []
#                 all_y = []
#                 # for index in range(1):
#                 #     position=np.argmax(hm_result[j][index])
#                 #     y,x=divmod(position, 64)
#                 #     all_x.append(x)
#                 #     all_y.append(y)
#                 #     plt.matshow(hm_result[j][index])
#                 plt.plot(all_x, all_y, 'r+')
#                 plt.plot(or_x, or_y, 'g+')
#                 leng = r_bbox.shape[1]
#                 for ii in range(leng // 2):
#                     rect = plt.Rectangle((r_bbox[j][ii * 2][0], r_bbox[j][ii * 2][1]),
#                                          r_bbox[j][ii * 2 + 1][0] - r_bbox[j][ii * 2][0],
#                                          r_bbox[j][ii * 2 + 1][1] - r_bbox[j][ii * 2][1],
#                                          linewidth=1,
#                                          edgecolor='r',
#                                          facecolor='none')
#
#                     plt.gca().add_patch(rect)
#                 leng = r_bbox.shape[1]
#                 for ii in range(leng // 2):
#                     rect = plt.Rectangle((r_bbox[j][ii * 2][0], r_bbox[j][ii * 2][1]),
#                                          r_bbox[j][ii * 2 + 1][0] - r_bbox[j][ii * 2][0],
#                                          r_bbox[j][ii * 2 + 1][1] - r_bbox[j][ii * 2][1],
#                                          linewidth=1,
#                                          edgecolor='g',
#                                          facecolor='none')
#
#                     plt.gca().add_patch(rect)
#                 plt.show()
#     coord.request_stop()
#     coord.join(threads)

# import matplotlib.pyplot as plt
# import numpy as np
#
# with tf.Session() as sess:  # 开始一个会话
#     init_op = tf.global_variables_initializer()
#     tf.local_variables_initializer().run()
#     sess.run(init_op)
#     # tf.local_variables_initializer().run()
#     # b_image, b_bbox, category_id, b_label, b_num_kpoints
#     b_image, b_bbox, category_id, b_label, _ = \
#         batch_samples_all_categories(1, '/root/dataset/tfrecord1/val/', False)
#     # b_image, b_bbox, b_label = \
#     #     batch_samples(1, '/root/dataset/tfrecord1/val/', False)
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
#                     # rect1 = plt.Rectangle((r_bbox[j][2][0], r_bbox[j][2][1]), r_bbox[j][3][0] - r_bbox[j][2][0],
#                     #                       r_bbox[j][3][1] - r_bbox[j][2][1], linewidth=1, edgecolor='r', facecolor='none')
#
#                     plt.gca().add_patch(rect)
#                 # plt.gca().add_patch(rect1)
#                 plt.show()
#     coord.request_stop()
#     coord.join(threads)
