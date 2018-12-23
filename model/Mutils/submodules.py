import tensorflow as tf


def batch_norm(input_images):
    # Batch Normalization批归一化
    # ((x-mean)/var)*gamma+beta
    # 输入通道维数
    # parms_shape=[input_images.get_shape()[-1]]
    # parms_shape=tf.shape(input_images)[-1]
    # print(parms_shape)
    # offset
    beta = tf.Variable(tf.constant(0.0, tf.float32), name='beta', dtype=tf.float32)
    # scale
    gamma = tf.Variable(tf.constant(1.0, tf.float32), name='gamma', dtype=tf.float32)
    # 为每个通道计算均值标准差
    mean, variance = tf.nn.moments(input_images, [0, 1, 2], name='moments')
    y = tf.nn.batch_normalization(input_images, mean, variance, beta, gamma, 0.001)
    y.set_shape(input_images.get_shape())

    return y


def batch_norm_relu(x):
    r_bn = batch_norm(x)
    r_bnr = tf.nn.relu(r_bn, name='relu')
    return r_bnr


def conv2(input_images, filter_size, stride, in_filters, out_filters, padding='SAME', xvaier=True):
    n = filter_size * filter_size * out_filters
    # 卷积核初始化
    if xvaier:

        weights = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)
                              ([filter_size, filter_size, in_filters, out_filters])
                              , name='weights')
    else:

        weights = tf.Variable(tf.random_normal(shape=[filter_size, filter_size, in_filters, out_filters],
                                               stddev=2.0 / n, dtype=tf.float32),
                              dtype=tf.float32,
                              name='weights')
    biases = tf.Variable(tf.constant(0.0, shape=[out_filters]), dtype=tf.float32, name='biases')
    r_conv = tf.nn.conv2d(input_images, weights, strides=stride, padding=padding)
    r_biases = tf.add(r_conv, biases)
    return r_biases


def pad_conv2(input_x, pad, filter_size, stride, in_filters, out_filters, xvaier=True):
    #
    # input_image = tf.Variable([[[[11, 21, 31], [41, 51, 61]], [[12, 23, 32], [43, 53, 63]]],
    #                            [[[1, 2, 3], [4, 5, 6]], [[14, 24, 34], [45, 55, 65]]]])
    # padding = tf.Variable([[0, 0], [0, 0], [3, 3], [3, 3]])
    if xvaier:

        weights = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)
                              ([filter_size, filter_size, in_filters, out_filters])
                              , name='weights')
    else:
        weights = tf.Variable(tf.random_normal(shape=[filter_size, filter_size, in_filters, out_filters],
                                               dtype=tf.float32),
                              dtype=tf.float32,
                              name='weights')
    padded = tf.pad(input_x, paddings=pad)

    conv_valid = tf.nn.conv2d(padded, weights, stride, padding='VALID')
    return conv_valid


def bottleneck_residual(input_images, stride, in_filters, out_filters, filter_size=None):
    orig_x = input_images
    mid_channels = int(out_filters // 2)  # 除法有些问题
    with tf.name_scope('r1'):
        with tf.name_scope('batch_norm_relu'):
            x = batch_norm_relu(input_images)
        with tf.name_scope('conv'):
            # input_images,filter_size,stride,in_filters,out_filters
            x = conv2(x, 1, stride, in_filters, mid_channels)
    with tf.name_scope('r2'):
        with tf.name_scope('batch_norm_relu'):
            x = batch_norm_relu(x)
        with tf.name_scope('conv'):
            x = conv2(x, 3, stride, mid_channels, mid_channels)
    with tf.name_scope('r3'):
        with tf.name_scope('batch_norm_relu'):
            x = batch_norm_relu(x)
        with tf.name_scope('conv'):
            x = conv2(x, 1, stride, mid_channels, out_filters)
    with tf.name_scope('skip'):
        if in_filters == out_filters:
            with tf.name_scope('identity'):
                orig_x = tf.identity(orig_x)
        else:
            with tf.name_scope('conv'):
                orig_x = conv2(orig_x, 1, stride, in_filters, out_filters)
    with tf.name_scope('sub_add'):
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


def hourglass(input_x, output_filters, nMoudel, n):
    # n表示hourglass的阶数
    orig_x = input_x
    with tf.name_scope('conv_road'):
        with tf.name_scope('down_sampling'):
            x = down_sampling(input_x, [1, 2, 2, 1], [1, 2, 2, 1])
        with tf.name_scope('pre_residual'):
            for i in range(nMoudel):
                with tf.name_scope('residual' + str(i + 1)):
                    x = bottleneck_residual(x, [1, 1, 1, 1], output_filters, output_filters)

        with tf.name_scope('hourglass' + str(n)):
            if n > 1:
                x = hourglass(x, output_filters, nMoudel, n - 1)
            else:
                x = bottleneck_residual(x, [1, 1, 1, 1], output_filters, output_filters)
        with tf.name_scope('back_residual'):
            for i in range(nMoudel):
                with tf.name_scope('residual' + str(i + 1)):
                    x = bottleneck_residual(x, [1, 1, 1, 1], output_filters, output_filters)
        with tf.name_scope('upsampling'):
            x = up_sampling(x)

    with tf.name_scope('skip_road'):
        with tf.name_scope('residual'):
            for i in range(nMoudel):
                with tf.name_scope('residual' + str(i + 1)):
                    orig_x = bottleneck_residual(orig_x, [1, 1, 1, 1], output_filters, output_filters)
    with tf.name_scope('sub_add'):
        y = tf.add(x, orig_x)
    return y


def lin(input_x, in_filters, out_filters):
    # 1*1卷积stride=1,卷积，bn，relu
    conv = conv2(input_x, 1, [1, 1, 1, 1], in_filters, out_filters)
    return batch_norm_relu(conv)


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')

    return op


def global_average_pooling(input):
    gap = tf.nn.avg_pool(input,
                         [1, input.shape[1], input.shape[2], 1],
                         [1, input.shape[1], input.shape[2], 1],
                         'VALID')
    return tf.reshape(gap, [gap.shape[0], -1])


def tail(r_lin, nFeats, ch, cell_size):
    return conv2(r_lin,
                 1,
                 [1, 1, 1, 1],
                 nFeats,
                 ch)


def tail_tsp(r_lin, nFeats, ch, cell_size):
    r_lin_x = tf.transpose(r_lin, perm=[0, 3, 2, 1])
    r_lin_y = tf.transpose(r_lin, perm=[0, 1, 3, 2])
    # print(nFeats // 2)
    x_conv = conv2(r_lin_x, 3, [1, 4, 1, 1], cell_size, nFeats // 2)
    y_conv = conv2(r_lin_y, 3, [1, 1, 4, 1], cell_size, nFeats // 2)
    c_conv = conv2(r_lin, 3, [1, 1, 1, 1], nFeats, nFeats)
    ct_conv = tf.concat([x_conv, y_conv, c_conv], axis=3)
    conv_down = conv2(ct_conv, 1, [1, 1, 1, 1], nFeats * 2, nFeats)
    return conv2(conv_down, 1, [1, 1, 1, 1], nFeats, ch)


def tail_tsp_self(r_lin, nFeats, ch, cell_size):
    r_lin_x = tf.transpose(r_lin, perm=[0, 3, 2, 1])
    r_lin_y = tf.transpose(r_lin, perm=[0, 1, 3, 2])
    x_conv = conv2(r_lin_x, 3, [1, 4, 1, 1], cell_size, nFeats // 2)
    y_conv = conv2(r_lin_y, 3, [1, 1, 4, 1], cell_size, nFeats // 2)
    c_conv = conv2(r_lin, 3, [1, 1, 1, 1], nFeats, nFeats)
    yolo_output_x = conv2(x_conv, 1, [1, 1, 1, 1], nFeats // 2, ch)
    yolo_output_y = conv2(y_conv, 1, [1, 1, 1, 1], nFeats // 2, ch)
    yolo_output_c = conv2(c_conv, 1, [1, 1, 1, 1], nFeats, ch)
    return yolo_output_x + yolo_output_y + yolo_output_c


def tail_conv(r_lin, nFeats, ch, cell_size):
    conv_1 = conv2(r_lin, 3, [1, 1, 1, 1], nFeats, nFeats)
    conv_2 = conv2(conv_1, 3, [1, 1, 1, 1], nFeats, nFeats)
    conv_3 = conv2(conv_2, 3, [1, 1, 1, 1], nFeats, nFeats * 2)
    conv_down = conv2(conv_3, 1, [1, 1, 1, 1], nFeats * 2, nFeats)
    return conv2(conv_down, 1, [1, 1, 1, 1], nFeats, ch)
