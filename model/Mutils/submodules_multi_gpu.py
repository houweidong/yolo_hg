import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import config as cfg


def batch_norm(input_images):
    # Batch Normalization批归一化
    # ((x-mean)/var)*gamma+beta
    # 输入通道维数
    # parms_shape=[input_images.get_shape()[-1]]
    # parms_shape=tf.shape(input_images)[-1]
    # print(parms_shape)
    # offset
    beta = tf.get_variable('beta', initializer=tf.constant(0.0, tf.float32), dtype=tf.float32)
    # scale
    gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, tf.float32), dtype=tf.float32)
    # 为每个通道计算均值标准差
    mean, variance = tf.nn.moments(input_images, [0, 1, 2], name='moments')
    y = tf.nn.batch_normalization(input_images, mean, variance, beta, gamma, 0.001)
    y.set_shape(input_images.get_shape())

    return y


def batch_norm_relu(x):
    r_bn = batch_norm(x)
    r_bnr = tf.nn.relu(r_bn, name='relu')
    return r_bnr


def conv2(input_images, filter_size, stride, in_filters, out_filters, l2, padding='SAME', xvaier=True):
    n = filter_size * filter_size * out_filters
    # 卷积核初始化
    if xvaier:

        weights = tf.get_variable('weights',
                                  shape=([filter_size, filter_size, in_filters, out_filters]),
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    else:

        weights = tf.get_variable('weights',
                                  initializer=tf.random_normal(
                                      shape=[filter_size, filter_size, in_filters, out_filters],
                                      stddev=2.0 / n, dtype=tf.float32),
                                  dtype=tf.float32)
    if l2:
        tf.add_to_collection('losses', l2(weights))
    biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[out_filters]), dtype=tf.float32)
    r_conv = tf.nn.conv2d(input_images, weights, strides=stride, padding=padding)
    r_biases = tf.add(r_conv, biases)
    return r_biases


def pad_conv2(input_x, pad, filter_size, stride, in_filters, out_filters, l2, xvaier=True):
    #
    # input_image = tf.Variable([[[[11, 21, 31], [41, 51, 61]], [[12, 23, 32], [43, 53, 63]]],
    #                            [[[1, 2, 3], [4, 5, 6]], [[14, 24, 34], [45, 55, 65]]]])
    # padding = tf.Variable([[0, 0], [0, 0], [3, 3], [3, 3]])
    if xvaier:

        weights = tf.get_variable('weights',
                                  shape=([filter_size, filter_size, in_filters, out_filters]),
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    else:

        weights = tf.get_variable('weights',
                                  initializer=tf.random_normal(
                                      shape=[filter_size, filter_size, in_filters, out_filters]),
                                  dtype=tf.float32)

    if l2:
        tf.add_to_collection('losses', l2(weights))
    padded = tf.pad(input_x, paddings=pad)

    conv_valid = tf.nn.conv2d(padded, weights, stride, padding='VALID')
    return conv_valid


def bottleneck_residual(input_images, stride, in_filters, out_filters, l2, filter_size=None):
    orig_x = input_images
    mid_channels = int(out_filters // 2)  # 除法有些问题
    with tf.variable_scope('r1'):
        with tf.variable_scope('batch_norm_relu'):
            x = batch_norm_relu(input_images)
        with tf.variable_scope('conv'):
            # input_images,filter_size,stride,in_filters,out_filters
            x = conv2(x, 1, stride, in_filters, mid_channels, l2)
    with tf.variable_scope('r2'):
        with tf.variable_scope('batch_norm_relu'):
            x = batch_norm_relu(x)
        with tf.variable_scope('conv'):
            x = conv2(x, 3, stride, mid_channels, mid_channels, l2)
    with tf.variable_scope('r3'):
        with tf.variable_scope('batch_norm_relu'):
            x = batch_norm_relu(x)
        with tf.variable_scope('conv'):
            x = conv2(x, 1, stride, mid_channels, out_filters, l2)
    with tf.variable_scope('skip'):
        if in_filters == out_filters:
            with tf.variable_scope('identity'):
                orig_x = tf.identity(orig_x)
        else:
            with tf.variable_scope('conv'):
                orig_x = conv2(orig_x, 1, stride, in_filters, out_filters, l2)
    with tf.variable_scope('sub_add'):
        # if in_filters!=out_filters:
        #     orig_x=conv2(orig_x,1,stride,in_filters,out_filters)
        result = tf.add(orig_x, x)
    return result


def down_sampling(x, ksize, strides, padding='VALID'):
    # 下采样
    return tf.nn.max_pool(x, ksize, strides, padding=padding, name='max_pool')


def up_sampling(x):
    # 反卷积实现
    # weights = tf.Variable(tf.random_normal(shape=[filter_size, filter_size, in_filters, out_filters],
    #                                        stddev=2.0 / n, dtype=tf.float32),
    #                       dtype=tf.float32,
    #                       name='weights')
    # tf.nn.conv2d_transpose(x,)
    # 最近邻插值实现
    # new_width=x.shape[1]*2
    # new_height=x.shape[2]*2
    y = tf.image.resize_nearest_neighbor(x, tf.shape(x)[1:3] * 2, name='upsampling')
    return y


def hourglass(input_x, output_filters, nMoudel, n, l2):
    # n表示hourglass的阶数
    orig_x = input_x
    with tf.variable_scope('conv_road'):
        with tf.variable_scope('down_sampling'):
            x = down_sampling(input_x, [1, 2, 2, 1], [1, 2, 2, 1])
        with tf.variable_scope('pre_residual'):
            for i in range(nMoudel):
                with tf.variable_scope('residual' + str(i + 1)):
                    x = bottleneck_residual(x, [1, 1, 1, 1], output_filters, output_filters, l2)

        with tf.variable_scope('hourglass' + str(n)):
            if n > 1:
                x = hourglass(x, output_filters, nMoudel, n - 1, l2)
            else:
                x = bottleneck_residual(x, [1, 1, 1, 1], output_filters, output_filters, l2)
        with tf.variable_scope('back_residual'):
            for i in range(nMoudel):
                with tf.variable_scope('residual' + str(i + 1)):
                    x = bottleneck_residual(x, [1, 1, 1, 1], output_filters, output_filters, l2)
        with tf.variable_scope('upsampling'):
            x = up_sampling(x)

    with tf.variable_scope('skip_road'):
        with tf.variable_scope('residual'):
            for i in range(nMoudel):
                with tf.variable_scope('residual' + str(i + 1)):
                    orig_x = bottleneck_residual(orig_x, [1, 1, 1, 1], output_filters, output_filters, l2)
    with tf.variable_scope('sub_add'):
        y = tf.add(x, orig_x)
    return y


def lin(input_x, in_filters, out_filters, l2):
    # 1*1卷积stride=1,卷积，bn，relu
    conv = conv2(input_x, 1, [1, 1, 1, 1], in_filters, out_filters, l2)
    return batch_norm_relu(conv)


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')

    return op


def global_average_pooling(x):
    gap = tf.nn.avg_pool(x,
                         [1, x.shape[1], x.shape[2], 1],
                         [1, x.shape[1], x.shape[2], 1],
                         'VALID')
    return tf.reshape(gap, [-1, x.shape[3]])


def tail(r_lin, csize_ch, l2):
    ch, _ = csize_ch
    # batch, r_lin_ch = r_lin.get_shape().as_list()[0], r_lin.get_shape().as_list()[3]
    r_lin_ch = 256
    return conv2(r_lin,
                 1,
                 [1, 1, 1, 1],
                 r_lin_ch,
                 ch,
                 l2)


def tail_tsp(r_lin, csize_ch, l2):
    ch, _ = csize_ch
    # shape = r_lin.get_shape().as_list()
    # batch, r_lin_csize, r_lin_ch = shape[0], shape[1], shape[3]
    r_lin_csize, r_lin_ch = cfg.IMAGE_SIZE // 4, 256
    r_lin_x = tf.transpose(r_lin, perm=[0, 3, 2, 1])
    r_lin_y = tf.transpose(r_lin, perm=[0, 1, 3, 2])
    # print(r_lin_ch // 2)
    x_conv = conv2(r_lin_x, 3, [1, 4, 1, 1], r_lin_csize, r_lin_ch // 2, l2)
    y_conv = conv2(r_lin_y, 3, [1, 1, 4, 1], r_lin_csize, r_lin_ch // 2, l2)
    c_conv = conv2(r_lin, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch, l2)
    ct_conv = tf.concat([x_conv, y_conv, c_conv], axis=3)
    conv_down = conv2(ct_conv, 1, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch, l2)
    return conv2(conv_down, 1, [1, 1, 1, 1], r_lin_ch, ch, l2)


def tail_tsp_self(r_lin, csize_ch, l2):
    ch, _ = csize_ch
    # shape = r_lin.get_shape().as_list()
    # batch, r_lin_csize, r_lin_ch = shape[0], shape[1], shape[3]
    r_lin_csize, r_lin_ch = cfg.IMAGE_SIZE // 4, 256
    r_lin_x = tf.transpose(r_lin, perm=[0, 3, 2, 1])
    r_lin_y = tf.transpose(r_lin, perm=[0, 1, 3, 2])
    x_conv = conv2(r_lin_x, 3, [1, 4, 1, 1], r_lin_csize, r_lin_ch // 2, l2)
    y_conv = conv2(r_lin_y, 3, [1, 1, 4, 1], r_lin_csize, r_lin_ch // 2, l2)
    c_conv = conv2(r_lin, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch, l2)
    yolo_output_x = conv2(x_conv, 1, [1, 1, 1, 1], r_lin_ch // 2, ch, l2)
    yolo_output_y = conv2(y_conv, 1, [1, 1, 1, 1], r_lin_ch // 2, ch, l2)
    yolo_output_c = conv2(c_conv, 1, [1, 1, 1, 1], r_lin_ch, ch, l2)
    return yolo_output_x + yolo_output_y + yolo_output_c


def tail_down4(r_lin, csize_ch, l2):
    ch, _ = csize_ch
    # _, r_lin_ch = r_lin.get_shape().as_list()[0], r_lin.get_shape().as_list()[3]
    r_lin_ch = 256
    with tf.variable_scope('conv1'):
        conv_1 = conv2(r_lin, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch, l2)
    with tf.variable_scope('conv2'):
        conv_2 = conv2(conv_1, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch, l2)
    with tf.variable_scope('conv3'):
        conv_3 = conv2(conv_2, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch * 2, l2)
    with tf.variable_scope('conv_down'):
        conv_down = conv2(conv_3, 1, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch, l2)
    with tf.variable_scope('result'):
        result = conv2(conv_down, 1, [1, 1, 1, 1], r_lin_ch, ch, l2)
    return result


def tail_down8(r_lin, csize_ch, l2):
    ch, _ = csize_ch
    # _, r_lin_ch = r_lin.get_shape().as_list()[0], r_lin.get_shape().as_list()[3]
    r_lin_ch = 256
    with tf.variable_scope('conv1'):
        conv_1 = conv2(r_lin, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch, l2)
    with tf.variable_scope('conv2'):
        conv_2 = conv2(conv_1, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch, l2)
    with tf.variable_scope('conv3'):
        conv_3 = conv2(conv_2, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch * 2, l2)
    with tf.variable_scope('down_sampling'):
        max_p = down_sampling(conv_3, [1, 2, 2, 1], [1, 2, 2, 1])
    with tf.variable_scope('conv_down'):
        conv_down = conv2(max_p, 1, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch, l2)
    with tf.variable_scope('result'):
        result = conv2(conv_down, 1, [1, 1, 1, 1], r_lin_ch, ch, l2)
    return result


def tail_down16(r_lin, csize_ch, l2):
    ch, _ = csize_ch
    # _, r_lin_ch = r_lin.get_shape().as_list()[0], r_lin.get_shape().as_list()[3]
    r_lin_ch = 256
    with tf.variable_scope('conv1'):
        conv_1 = conv2(r_lin, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch, l2)
    with tf.variable_scope('conv2'):
        conv_2 = conv2(conv_1, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch, l2)
    with tf.variable_scope('conv3'):
        conv_3 = conv2(conv_2, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch * 2, l2)
    with tf.variable_scope('down_sampling'):
        max_p = down_sampling(conv_3, [1, 2, 2, 1], [1, 2, 2, 1])
    with tf.variable_scope('cv_1'):
        cv_1 = conv2(max_p, 3, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch * 2, l2)
    with tf.variable_scope('cv_2'):
        cv_2 = conv2(cv_1, 3, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch * 2, l2)
    with tf.variable_scope('cv_3'):
        cv_3 = conv2(cv_2, 3, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch * 4, l2)
    with tf.variable_scope('down_sampling1'):
        max_p = down_sampling(cv_3, [1, 2, 2, 1], [1, 2, 2, 1])
    with tf.variable_scope('conv_down'):
        conv_down = conv2(max_p, 1, [1, 1, 1, 1], r_lin_ch * 4, r_lin_ch, l2)
    with tf.variable_scope('result'):
        result = conv2(conv_down, 1, [1, 1, 1, 1], r_lin_ch, ch, l2)
    return result


def tail_down16_v2(r_lin, csize_ch, l2):
    ch, _ = csize_ch
    # _, r_lin_ch = r_lin.get_shape().as_list()[0], r_lin.get_shape().as_list()[3]
    r_lin_ch = 256
    with tf.variable_scope('down1_cv1'):
        down1_cv1 = conv2(r_lin, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch * 2, l2)
    with tf.variable_scope('down1_cv2'):
        down1_cv2 = conv2(down1_cv1, 1, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch, l2)
    with tf.variable_scope('down1_cv3'):
        down1_cv3 = conv2(down1_cv2, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch * 2, l2)
    with tf.variable_scope('down_sampling1'):
        down1_max_p = down_sampling(down1_cv3, [1, 2, 2, 1], [1, 2, 2, 1])

    with tf.variable_scope('down2_cv1'):
        down2_cv1 = conv2(down1_max_p, 3, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch * 4, l2)
    with tf.variable_scope('down2_cv2'):
        down2_cv2 = conv2(down2_cv1, 1, [1, 1, 1, 1], r_lin_ch * 4, r_lin_ch * 2, l2)
    with tf.variable_scope('down2_cv3'):
        down2_cv3 = conv2(down2_cv2, 3, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch * 4, l2)
    with tf.variable_scope('down_sampling2'):
        down2_max_p = down_sampling(down2_cv3, [1, 2, 2, 1], [1, 2, 2, 1])

    with tf.variable_scope('end_cv1'):
        end_cv1 = conv2(down2_max_p, 3, [1, 1, 1, 1], r_lin_ch * 4, r_lin_ch * 4, l2)
    with tf.variable_scope('end_cv2'):
        end_cv2 = conv2(end_cv1, 3, [1, 1, 1, 1], r_lin_ch * 4, r_lin_ch * 4, l2)
    with tf.variable_scope('end_cv3'):
        end_cv3 = conv2(end_cv2, 3, [1, 1, 1, 1], r_lin_ch * 4, r_lin_ch * 4, l2)
    with tf.variable_scope('end_cv4'):
        end_cv4 = conv2(end_cv3, 3, [1, 1, 1, 1], r_lin_ch * 4, r_lin_ch * 4, l2)
    with tf.variable_scope('result'):
        result = conv2(end_cv4, 1, [1, 1, 1, 1], r_lin_ch * 4, ch, l2)
    return result


def tail_conv_deep(r_lin, csize_ch, l2):
    ch, _ = csize_ch
    # _, r_lin_ch = r_lin.get_shape().as_list()[0], r_lin.get_shape().as_list()[3]
    r_lin_ch = 256
    conv_1 = conv2(r_lin, 3, [1, 1, 1, 1], r_lin_ch, r_lin_ch * 2, l2)
    conv_2 = conv2(conv_1, 3, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch * 2, l2)
    conv_3 = conv2(conv_2, 3, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch * 2, l2)
    conv_down = conv2(conv_3, 1, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch, l2)
    return conv2(conv_down, 1, [1, 1, 1, 1], r_lin_ch, ch, l2)


def tail_conv_deep_fc(r_lin, csize_ch, l2):
    ch, cell_size = csize_ch
    # batch, r_lin_ch = r_lin.get_shape().as_list()[0], r_lin.get_shape().as_list()[3]
    r_lin_ch = 256
    with tf.variable_scope('tail_residual1'):
        r1 = bottleneck_residual(r_lin, [1, 1, 1, 1], r_lin_ch, r_lin_ch * 2, l2)
    with tf.variable_scope('tail_down_sampling1'):
        ds = down_sampling(r1, [1, 2, 2, 1], [1, 2, 2, 1])  # 32 * 32 * 512
    with tf.variable_scope('tail_residual2'):
        r1 = bottleneck_residual(ds, [1, 1, 1, 1], r_lin_ch * 2, r_lin_ch * 4, l2)
    with tf.variable_scope('tail_down_sampling2'):
        ds = down_sampling(r1, [1, 2, 2, 1], [1, 2, 2, 1])  # 16 * 16 * 1024
    with tf.variable_scope('tail_residual3'):
        r1 = bottleneck_residual(ds, [1, 1, 1, 1], r_lin_ch * 4, r_lin_ch * 4, l2)
    max_p = global_average_pooling(r1)
    fc = slim.fully_connected(max_p, cell_size * cell_size * ch, activation_fn=None, scope='fc')
    return tf.reshape(fc, [-1, cell_size, cell_size, ch])
