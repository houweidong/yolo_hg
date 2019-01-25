# import numpy as np
# import torch as t
# from torch.autograd import Variable as V
# from torch import nn

# a = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
#      [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]
# b = np.reshape(a, -1)
# c = np.reshape(b, (4, 2, 3))
# d = np.transpose(c, (1, 2, 0))
# print(d.shape)
# print(np.reshape(d[:, :, 0], (2, 3, 1)))


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         param1 = nn.Parameter(t.rand(3, 3))
#         submodel1 = nn.Linear(3, 4)
#
#     def forward(self, input):
#         x = param1@input@(input.view(1, 3))
#         x = submodel1(x)
#         return x
#
# net = Net()
# print(net)
# a = t.Tensor([1, 2])
# b = net(a)
# print(b)

# bn = nn.BatchNorm1d(2)
# input = V(t.rand(3, 2), requires_grad=True)
# output = bn(input)
# print(bn._buffers)

# import torch as t
# from torch import nn
#
# a = t.Tensor()
# print(a)
# module = nn.Module()
# module.param = nn.Parameter(t.ones(2, 2))
# submodule1 = nn.Linear(2, 2)
# submodule2 = nn.Linear(2, 2)
# module_list = [submodule1, submodule2]
# module.submodules = module_list
# module_list1 = nn.ModuleList(module_list)
# module.submodules1 = module_list1
#
#
# print(module.__dict__)
# print(module._modules)
# print(module.__dict__.get('submodules1'))


#
# filter()

# x = t.arange(0, 9, 0.1)
# y = (x ** 2) / 9
# vis.updateTrace(X=x, Y=y, win='polynomial', name='this is')

# all_data = t.load('all.pth')
# print(all_data.keys())
# model = Net()
# model.load_state_dict(all_data['model'])
# print(model.state_dict())

# from sklearn.metrics import average_precision_score
#
# # y_pred是预测标签
# y_pred, y_true = np.array([0.9993857145309448, 0.9993764758110046, 0.9993315935134888,
#                   0.6958107948303223, 0.99920254945755, 0.998860239982605,
#                   0.983239471912384, 0.9813326597213745, 0.9993232488632202]), \
#                  np.array([1, 1, 1, 0, 1, 1, 0, 0, 1])
# print(average_precision_score(y_true=y_true, y_score=y_pred, average='weighted'))
# print(average_precision_score(y_true=y_true, y_score=y_pred, average='macro'))
# print(average_precision_score(y_true=y_true, y_score=y_pred, average='micro'))
# print(average_precision_score(y_true=y_true, y_score=y_pred))

import tensorflow as tf
# state = tf.Variable(0.0,dtype=tf.float32)
import numpy as np

# def tester(start):
#     def nested(label):
#         nested.state = start
#         print(label, nested.state)
#         nested.state += 1
#         print(nested.state)
#
#     return nested
#
# F = tester(0)
# F('SPAN')


# import numpy as np
# from utils.timer import Timer
# a = np.array([1, 2, 3])
# b = np.array([1.0, 3, 3.0])
# print(np.equal(a, b))
# import argparse
# import logging
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--log_dir', default='xiaolunwen/dir', type=str)
# args = parser.parse_args()
#
# if args.log_dir is not None:
#     logging.basicConfig(filename=args.log_dir, level=logging.INFO)
# else:
#     logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()
# # logger.addHandler(logging.StreamHandler())
# logger = logger.info
# logger("wode")
# import json
# import os
# import contextlib2
# import tensorflow as tf
# from utils import config as cfg
#
#
#
# det_annotations_file = '/root/dataset/annotations_trainval2017/annotations/instances_val2017.json'
# kp_annotations_file= '/root/dataset/annotations_trainval2017/annotations/person_keypoints_val2017.json'
# with contextlib2.ExitStack() as tf_record_close_stack, \
#         tf.gfile.GFile(det_annotations_file, 'r') as det_fid, \
#         tf.gfile.GFile(kp_annotations_file, 'r') as kp_fid:
#     # json file
#     det_data = json.load(det_fid)
#     kp_data = json.load(kp_fid)
#
#     # det_data images should include kp_data images
#     images = det_data['images']
#     annotations_index = {}
#     print(
#         'Found {:<5} groundtruth annotations. Building annotations index.'
#             .format(len(det_data['annotations'])))
#     print(
#         'Found {:<5} groundtruth annotations. Building annotations index.'
#             .format(len(kp_data['annotations'])))
#
#     for annotation in det_data['annotations']:
#         image_id = annotation['image_id']
#         # num_keypoints = annotation['num_keypoints']
#         if image_id not in annotations_index:
#             annotations_index[image_id] = {}
#         if annotation['category_id'] != 1:
#             annotation['category_id'] = 0
#         annotation['keypoints'] = [0] * 51
#         annotation['num_keypoints'] = 0
#         annotations_index[image_id][annotation['id']] = annotation
#
#     # for image in images:
#     #     if image['id'] not in annotations_index:
#     #         print(image['file_name'])
#
#     for annotation_kp in kp_data['annotations']:
#         image_id = annotation_kp['image_id']
#         # if image_id not in annotations_index:
#         #     print('there has images exists in kp_annotations but not exists in det_annotations')
#         #     continue
#         # if annotation_kp['id'] not in annotations_index[image_id]:
#         #     print('#there has annotations exists in kp_annotations but not exists in det_annotations')
#         #     continue
#         annotations_index[image_id][annotation_kp['id']] = annotation_kp
#
#     # print('len_anno_index',len(annotations_index))
#     missing_annotation_count = 0
#     for image in images:
#         image_id = image['id']
#         if image_id not in annotations_index:
#             missing_annotation_count += 1
#             annotations_index[image_id] = []
#     print('{} images are missing annotations.'.format(missing_annotation_count))
# import utils.config as cfg
# import numpy as np
#
# a = np.array([1.0, 0.0, 1.0, 0.0])
# b = np.array([[1, 2],[3, 4],[5, 6],[7, 8]])
# a = a.astype(np.bool)
# print(a)
# print(b[a])

# import numpy as np
# import utils.config as cfg
#
# box_hm_sigma = 2
# box_hm_level = cfg.BOX_HOT_MAP_LEVEL
# box_hm_prob_size = 2 * box_hm_level + 1
# box_hm_gaussian = True
#
# dmax = box_hm_level * pow(2, 0.5)
# h_w = box_hm_level * 2 + 1
# col = np.reshape(np.array([np.arange(h_w)] * h_w), (h_w, h_w))
# row = np.transpose(col)
# print(row)
# print(col)
# center = box_hm_level
# # print(d)
# if not box_hm_gaussian:
#     d = np.sqrt(np.square(row - center) + np.square(col - center))
#     prob = np.ones((h_w, h_w)) - 0.9 / dmax * d
# else:
#     prob = np.exp(-1 * (np.square(row - center) +
#                         np.square(col - center)) / box_hm_sigma ** 2)
# print(prob)
# cell_size = 6
# y_ind = 2
# x_ind = 2
# lt, rt, tp, dn = y_ind - box_hm_level, y_ind + box_hm_level + 1, \
#                  x_ind - box_hm_level, x_ind + box_hm_level + 1
# left, right, top, down = max(lt, 0), min(rt, cell_size), \
#                          max(tp, 0), min(dn, cell_size)
# lt_mg, rt_mg, tp_mg, dn_mg = left - lt, right - rt, top - tp, down - dn
# prob = prob[lt_mg:box_hm_prob_size + rt_mg,
#        tp_mg:box_hm_prob_size + dn_mg]
# print(prob)
#
# aa1 = np.random.randint(-5, 5, (6, 6, 5)).astype(np.float32)
# # print(aa)
# aa = aa1[:, :, 0]
# bb = aa[left:right, top:down]
# r_w = bb > prob
# cc = np.where(r_w, bb, prob)
# mybbox = np.array([1, 1, 1, 1])
# bbox = np.where(r_w[:, :, np.newaxis], aa1[left:right, top:down, 1:], mybbox[np.newaxis, np.newaxis, :])
# aa[left:right, top:down] = cc
# print(aa)
# print(bbox)
# names = {'ADD_YOLO_POSITION', 'LOSS_FACTOR', 'OBJECT_SCALE', 'NOOBJECT_SCALE',
#          'COORD_SCALE', 'BOEX_FOCAL_LOSS'}
# print(names)
# names['ADD_YOLO_POSITION'] = 1
# print(names)

# a = 'True'
# print(bool(a))
# import collections
# # values = collections.OrderedDict()
# # # values = {'ADD_YOLO_POSITION': None, 'LOSS_FACTOR': None, 'OBJECT_SCALE': None,
# # #           'NOOBJECT_SCALE': None, 'COORD_SCALE': None, 'BOEX_FOCAL_LOSS': None}
# # print(values)
# values = collections.OrderedDict()
# keys = ['ADD_YOLO_POSITION', 'LOSS_FACTOR', 'OBJECT_SCALE',
#         'NOOBJECT_SCALE', 'COORD_SCALE', 'BOEX_FOCAL_LOSS']
# values.fromkeys(keys)
# values['ADD_YOLO_POSITION'] = 1
# values['LOSS_FACTOR'] = 2
# strings = ''
# for i, value in values.items():
#     strings += '{}: {}  '.format(i, value)
# print(strings)
import os
from functools import reduce


# files = os.listdir(dirpath)
# files.sort()
# files.sort(key=lambda x: int(x[:-4]))

# model_start = 'hg_yolo'
# rootdir = 'log/'
# list1 = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
# list1.sort()
# for path in list1:
#     model_dir = os.path.join(rootdir, path)
#     models = os.listdir(model_dir)
#     models = filter(lambda x: x.startswith(model_start), models)
#     models = list(set(map(lambda x: x.split('.')[0], models)))
#     models.sort(key=lambda x: int(x[8:]))
#
#     for model in models:
#         print(model)

# a = [1, 10, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
# b = set(a)
#
# print(b)
# print(id(a))
# print(id(b))
# import numpy as np
#
# cell_size = 64
# image_size = 256
#
# xmin, xmax, ymin, ymax = 0.0, 256.0, 0.0, 256.0
# x1, x2, y1, y2 = map(lambda x: x / 2, (xmin, xmax, ymin, ymax))
# x11, x22, y11, y22 = map(lambda x: x / 4, (xmin, xmax, ymin, ymax))
# boxes = [[xmin, xmax, ymin, ymax], [x1, x2, y1, y2], [x11, x22, y11, y22]]
# prob = []
# for box in boxes:
#     l, r, t, d = map(lambda x: int(x * cell_size / image_size), box)
#     grid_w, grid_h = r + 1 - l, d + 1 - t
#     imag_w, imag_h = box[1] + 1 - box[0], box[3] + 1 - box[2]
#     factor_all = 1 / np.power(imag_h * imag_w, 1/2) * 5000
#     sigma_w, sigma_h = factor_all * imag_w / 2, factor_all * imag_h / 2
#
#     col = np.reshape(np.array([np.arange(grid_w)] * grid_h), (grid_h, grid_w))
#     row = np.transpose(np.reshape(np.array([np.arange(grid_h)] * grid_w), (grid_w, grid_h)))
#     center_col, center_row = (grid_w - 1) / 2, (grid_h - 1) / 2
#     prob_sub = np.exp(-1 * ((np.square(col - center_col)) / sigma_w ** 2
#                         + (np.square(row - center_row)) / sigma_h ** 2))
#     print(prob_sub[int(center_col), int(center_row):])
#     prob.append(prob_sub)
# print(prob)

#
# me = '../log/20_1_100_conv_fc'
# a = me.split('/')[2] + '  '
# print(a, me)

# import sys
# print(type(sys.path))
# for p in sys.path:
#     print(p)
# print("*"*20)
# sys.path.append('D:\code')
# for p in sys.path:
#     print(p)
# h = 3
# w = 7
# a = np.random.randn(2, h, w)
# b = a.reshape([2, -1])
# print(b)
# c = np.argmax(b, 1)
# d = np.max(b, 1)
# print(d)
# row, col = np.divmod(c, w)
# row = np.where(d>1.5, row, -1)
# col = np.where(d>1.5, col, -1)
# print(row, col)

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/mnist/", one_hot=True)

num_gpus = 2
num_steps = 1000
learning_rate = 0.001
batch_size = 1000
display_step = 10

num_input = 784
num_classes = 10


def conv_net_with_layers(x, is_training, dropout=0.75):
    with tf.variable_scope("ConvNet", reuse=tf.AUTO_REUSE):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = tf.layers.conv2d(x, 12, 5, activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 24, 3, activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 100)
        x = tf.layers.dropout(x, rate=dropout, training=is_training)
        out = tf.layers.dense(x, 10)
        out = tf.nn.softmax(out) if not is_training else out
    return out


def conv_net(x, is_training):
    # "updates_collections": None is very import ,without will only get 0.10
    batch_norm_params = {"is_training": is_training, "decay": 0.9, "updates_collections": None}
    # ,'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ]
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        with tf.variable_scope("ConvNet", reuse=tf.AUTO_REUSE):
            x = tf.reshape(x, [-1, 28, 28, 1])
            net = slim.conv2d(x, 6, [5, 5], scope="conv_1")
            net = slim.conv2d(net, 6, [5, 5], scope="conv_11")
            net = slim.max_pool2d(net, [2, 2], scope="pool_1")
            net = slim.conv2d(net, 12, [5, 5], scope="conv_2")
            net = slim.max_pool2d(net, [2, 2], scope="pool_2")
            net = slim.flatten(net, scope="flatten")
            net = slim.fully_connected(net, 100, scope="fc")
            net = slim.dropout(net, is_training=is_training)
            net = slim.fully_connected(net, num_classes, scope="prob", activation_fn=None, normalizer_fn=None)
            return net


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.device("/cpu:0"):
        global_step = tf.train.get_or_create_global_step()
        tower_grads = []
        X = tf.placeholder(tf.float32, [None, num_input])
        Y = tf.placeholder(tf.float32, [None, num_classes])
        opt = tf.train.AdamOptimizer(learning_rate)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(1):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("tower_%d" % i):
                        _x = X[i * batch_size:(i + 1) * batch_size]
                        _y = Y[i * batch_size:(i + 1) * batch_size]
                        logits = conv_net(_x, True)
                        tf.get_variable_scope().reuse_variables()
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=logits))
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)
                        if i == 0:
                            logits_test = conv_net(_x, False)
                            correct_prediction = tf.equal(tf.argmax(logits_test, 1), tf.argmax(_y, 1))
                            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        grads = average_gradients(tower_grads)
        train_op = opt.apply_gradients(grads)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(1, num_steps + 1):
                batch_x, batch_y = mnist.train.next_batch(batch_size * num_gpus)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % 10 == 0 or step == 1:
                    loss_value, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
                    print("Step:" + str(step) + ":" + str(loss_value) + " " + str(acc))
            print("Done")
            print("Testing Accuracy:",
                  np.mean([sess.run(accuracy, feed_dict={X: mnist.test.images[i:i + batch_size],
                                                         Y: mnist.test.labels[i:i + batch_size]}) for i in
                           range(0, len(mnist.test.images), batch_size)]))


def train_single():
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    logits = conv_net(X, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
    opt = tf.train.AdamOptimizer(learning_rate)
    train_op = opt.minimize(loss)
    logits_test = conv_net(X, False)
    correct_prediction = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(1, num_steps + 1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                loss_value, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Step:" + str(step) + ":" + str(loss_value) + " " + str(acc))
        print("Done")
        print("Testing Accuracy:", np.mean([sess.run(accuracy, feed_dict={X: mnist.test.images[i:i + batch_size],
                                                                          Y: mnist.test.labels[i:i + batch_size]}) for i
                                            in
                                            range(0, len(mnist.test.images), batch_size)]))


if __name__ == "__main__":
    # train_single()
    train()
# def funcdeep():
#     print('funcdeep-----------------')
#     vfuncno = tf.get_variable("v_funcdeep", [1])
#     vfuncno1 = tf.get_variable("v_funcdeep1", [1])
#     vfuncno2 = vfuncno + vfuncno1
#
#     print(vfuncno.name)
#     print(vfuncno1.name)
#     print(vfuncno2.name)
#
#
# def func():
#     print('func-----------------')
#     print("scope inner:", tf.get_variable_scope())
#     with tf.variable_scope(tf.get_variable_scope()):
#         print("scope inner:", scope)
#         vfunc = tf.get_variable("v_func", [1])
#         print(vfunc.name)
#
#
# def func_no():
#     print('funcno-----------------')
#     vfuncno = tf.get_variable("v_funcno", [1])
#     vfuncno1 = tf.get_variable("v_funcno1", [1])
#     vfuncno2 = vfuncno + vfuncno1
#
#     print(vfuncno.name)
#     print(vfuncno1.name)
#     print(vfuncno2.name)
#     funcdeep()
#
#
# with tf.name_scope("root1") as scope:
#     print(tf.get_variable_scope())
#     print(tf.get_variable_scope().reuse)
#     print(tf.get_variable_scope().name)
#
#     print(".....")
#
#     v1 = tf.get_variable("v1", [1])
#     v2 = tf.Variable(tf.constant(1.0, shape=[1]), name="v")
#
#     v4 = v1 + v2
#     print("scope outer:", scope)
#     func()
#     func_no()
#
#     print(v1.name)
#     print(v2.name)
#     print(v4.name)
#     with tf.name_scope("root"):
#         v3 = tf.ones([2, 3], name='v')
#         print(v3.name)
#
# print('-----------------')
# with tf.name_scope("root"):
#     print(tf.get_variable_scope())
#     print(tf.get_variable_scope().reuse)
#     print(tf.get_variable_scope().name)
#
#     print(".....")
#
#     v1 = tf.get_variable("v", [1])
#     v2 = tf.Variable(tf.constant(1.0, shape=[1]), name="v")
#     print(v1.name)
#     print(v2.name)
#
#
#
# print('-----------------')
# for v in tf.global_variables():
#     print(v.name)


def tower_loss(scope, images, labels):
    """
    计算当前tower的损失
    这里的损失包括最后的损失和weight的L2正则损失 具体可以
    :param scope: 当前空间名
    :param images: 输入的图像
    :param labels: 图像的label
    :return: 总的loss
    """
    logits = cifar10.inference(images)
    _ = cifar10.loss(logits, labels)
    # 获得losses集合中的所有损失
    losses = tf.get_collection('losses', scope)
    # 将所有损失加和
    total_loss = tf.add_n(losses, name='total_loss')
    # 将loss记录到summary中
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)
    return total_loss

