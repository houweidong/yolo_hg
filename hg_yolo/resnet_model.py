import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
import hg_yolo.config as cfg
from tensorflow.examples.tutorials.mnist import  input_data

"""
残差网络
"""

nMoudel=1#hourglass 中residual 模块的数量
nStack=2#hourglass 堆叠的层数
nFeats=256 #hourglass 中特征图的数量
nPoint=17#关键点个数

def batch_norm(input_images):
    # Batch Normalization批归一化
    # ((x-mean)/var)*gamma+beta
    #输入通道维数
    #parms_shape=[input_images.get_shape()[-1]]
    #parms_shape=tf.shape(input_images)[-1]
    #print(parms_shape)
    #offset
    beta=tf.Variable(tf.constant(0.0,tf.float32),name='beta',dtype=tf.float32)
    #scale
    gamma=tf.Variable(tf.constant(1.0,tf.float32),name='gamma',dtype=tf.float32)
    #为每个通道计算均值标准差
    mean,variance=tf.nn.moments(input_images, [0, 1, 2], name='moments')
    y=tf.nn.batch_normalization(input_images,mean,variance,beta,gamma,0.001)
    y.set_shape(input_images.get_shape())

    return y


def batch_norm_relu(x):
    r_bn=batch_norm(x)
    r_bnr=tf.nn.relu(r_bn,name='relu')
    return  r_bnr


def conv2(input_images,filter_size,stride,in_filters,out_filters,padding='SAME',xvaier=True):

    n=filter_size*filter_size*out_filters
    #卷积核初始化
    if xvaier:

        weights=tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)
                            ([filter_size,filter_size,in_filters,out_filters])
                            ,name = 'weights')
    else:

        weights=tf.Variable(tf.random_normal(shape=[filter_size,filter_size,in_filters,out_filters],
                                         stddev=2.0/n,dtype=tf.float32),
                        dtype=tf.float32,
                        name='weights')
    biases=tf.Variable(tf.constant(0.0,shape=[out_filters]),dtype=tf.float32,name='biases')
    r_conv=tf.nn.conv2d(input_images,weights,strides=stride,padding=padding)
    r_biases=tf.add(r_conv,biases)
    return r_biases


def pad_conv2(input_x,pad,filter_size,stride,in_filters,out_filters,xvaier=True):
    #
    # input_image = tf.Variable([[[[11, 21, 31], [41, 51, 61]], [[12, 23, 32], [43, 53, 63]]],
    #                            [[[1, 2, 3], [4, 5, 6]], [[14, 24, 34], [45, 55, 65]]]])
    # padding = tf.Variable([[0, 0], [0, 0], [3, 3], [3, 3]])
    if xvaier:

        weights=tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)
                            ([filter_size,filter_size,in_filters,out_filters])
                            ,name = 'weights')
    else:
        weights = tf.Variable(tf.random_normal(shape=[filter_size, filter_size, in_filters, out_filters],
                                           dtype=tf.float32),
                          dtype=tf.float32,
                          name='weights')
    padded=tf.pad(input_x,paddings=pad)

    conv_valid=tf.nn.conv2d(padded,weights,stride,padding='VALID')
    return conv_valid


def bottleneck_residual(input_images,stride,in_filters,out_filters,filter_size=None):
    orig_x=input_images
    mid_channels=int(out_filters//2)#除法有些问题
    with tf.name_scope('r1'):
        with tf.name_scope('batch_norm_relu'):
            x=batch_norm_relu(input_images)
        with tf.name_scope('conv'):
            #input_images,filter_size,stride,in_filters,out_filters
            x=conv2(x,1,stride,in_filters,mid_channels)
    with tf.name_scope('r2'):
        with tf.name_scope('batch_norm_relu'):
            x=batch_norm_relu(x)
        with tf.name_scope('conv'):
            x=conv2(x,3,stride,mid_channels,mid_channels)
    with tf.name_scope('r3'):
        with tf.name_scope('batch_norm_relu'):
            x=batch_norm_relu(x)
        with tf.name_scope('conv'):
            x=conv2(x,1,stride,mid_channels,out_filters)
    with tf.name_scope('skip'):
        if in_filters==out_filters:
            with tf.name_scope('identity'):
                orig_x=tf.identity(orig_x)
        else:
            with tf.name_scope('conv'):
                orig_x=conv2(orig_x,1,stride,in_filters,out_filters)
    with tf.name_scope('sub_add'):
        # if in_filters!=out_filters:
        #     orig_x=conv2(orig_x,1,stride,in_filters,out_filters)
        result=tf.add(orig_x,x)
    return result


def down_sampling(x,ksize,strides,padding='VALID'):

    #下采样
    return tf.nn.max_pool(x,ksize,strides,padding=padding,name='max_pool')


def up_sampling(x):
    #反卷积实现
    # weights = tf.Variable(tf.random_normal(shape=[filter_size, filter_size, in_filters, out_filters],
    #                                        stddev=2.0 / n, dtype=tf.float32),
    #                       dtype=tf.float32,
    #                       name='weights')
    # tf.nn.conv2d_transpose(x,)
    #最近邻插值实现
    # new_width=x.shape[1]*2
    # new_height=x.shape[2]*2
    y=tf.image.resize_nearest_neighbor(x,tf.shape(x)[1:3]*2,name='upsampling')
    return y


def hourglass(input_x,output_filters,n):

    #n表示hourglass的阶数
    orig_x=input_x
    with tf.name_scope('conv_road'):
        with tf.name_scope('down_sampling'):
            x=down_sampling(input_x,[1,2,2,1],[1,2,2,1])
        with tf.name_scope('pre_residual'):
            for i in range(nMoudel):
                with tf.name_scope('residual'+str(i+1)):
                    x=bottleneck_residual(x,[1,1,1,1],output_filters,output_filters)

        with tf.name_scope('hourglass'+str(n)):
            if n>1:
                x=hourglass(x,output_filters,n-1)
            else:
                x=bottleneck_residual(x,[1,1,1,1],output_filters,output_filters)
        with tf.name_scope('back_residual'):
            for i in range(nMoudel):
                with tf.name_scope('residual'+str(i+1)):
                    x=bottleneck_residual(x,[1,1,1,1],output_filters,output_filters)
        with tf.name_scope('upsampling'):
            x=up_sampling(x)

    with tf.name_scope('skip_road'):
        with tf.name_scope('residual'):
            for i in range(nMoudel):
                with tf.name_scope('residual'+str(i+1)):
                    orig_x=bottleneck_residual(orig_x,[1,1,1,1],output_filters,output_filters)
    with tf.name_scope('sub_add'):
        y=tf.add(x,orig_x)
    return y


def lin(input_x,in_filters,out_filters):
    #1*1卷积stride=1,卷积，bn，relu
    conv=conv2(input_x,1,[1,1,1,1],in_filters,out_filters)
    return batch_norm_relu(conv)


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op


#def model(input_x, is_training):
def model(input_x,
          shape,
          alpha,
          keep_prob,
          is_training=True,
          add_yolo_position="tail"):  # add yolo position only support tail and middle
    #conv=conv2(input_x,7,[1,2,2,1])
    batch_size, cell_hight, cell_width, ch = shape
    with tf.name_scope('conv_pad3'):
        cp=pad_conv2(input_x,[[0,0],[3,3],[3,3],[0,0]],7,[1,2,2,1],3,64)
    with tf.name_scope('batch_norm_relu'):
        bn=batch_norm_relu(cp)
    with tf.name_scope('residual1'):
        r1=bottleneck_residual(bn,[1,1,1,1],64,128)
    with tf.name_scope('down_sampling'):
        ds=down_sampling(r1,[1,2,2,1],[1,2,2,1])
    with tf.name_scope('residual2'):
        r2=bottleneck_residual(ds,[1,1,1,1],128,128)
    with tf.name_scope('residual3'):
        r3=bottleneck_residual(r2,[1,1,1,1],128,nFeats)

    output, yolo_output = None, None
    # hourglass 的输入
    h_input = r3
    for n in range(nStack):
        with tf.name_scope('hourglass'+str(n+1)):
            h1 = hourglass(h_input, nFeats, 4)
        residual=h1
        for i in range(nMoudel):
            with tf.name_scope('residual' + str(i + 1)):
                residual = bottleneck_residual(residual, [1, 1, 1, 1], nFeats, nFeats)
        with tf.name_scope('lin'):
            r_lin=lin(residual,nFeats,nFeats)

            # add yolo_head in the tail of hg_net
            if n == nStack-1 and add_yolo_position == "tail":
                yolo_output = conv2(r_lin,
                                    1,
                                    [1, 1, 1, 1],
                                    nFeats,
                                    ch)


        with tf.name_scope('conv_same'):
            output=conv2(r_lin,1,[1,1,1,1],nFeats,nPoint,padding='VALID')#特征图输出
        if n<(nStack-1):
            #print(n)
            with tf.name_scope('next_input'):
                c_output=conv2(output,1,[1,1,1,1],nPoint,nFeats)#卷积的输出
                h_input=tf.add(h_input,tf.add(r_lin,c_output))

    #output=tf.reshape(output,(-1,16,64,64),name='output')
    output=tf.transpose(output,[0,3,1,2],name='output')#transpose和reshape结果是不一样的


    # with tf.name_scope('yolo_head'):
    #     yolo_max0 = down_sampling(r3, [1, 2, 2, 1], [1, 2, 2, 1])  # 32*32*256
    #     yolo_convblock_f1 = bottleneck_residual(yolo_max0, [1, 1, 1, 1], 256, 256)
    #     yolo_convblock_s1 = bottleneck_residual(yolo_convblock_f1, [1, 1, 1, 1], 256, 512)
    #     yolo_conv1 = conv2(yolo_convblock_s1, 3, [1, 1, 1, 1], 512, 512)
    #     yolo_max1 = down_sampling(yolo_conv1, [1, 2, 2, 1], [1, 2, 2, 1])  # 16*16*512
    #
    #     yolo_convblock_f2 = bottleneck_residual(yolo_max1, [1, 1, 1, 1], 512, 512)
    #     yolo_convblock_s2 = conv2(yolo_convblock_f2, 3, [1, 1, 1, 1], 512, 1024)
    #     yolo_conv2 = conv2(yolo_convblock_s2, 3, [1, 1, 1, 1], 1024, 1024)
    #     yolo_conv_stride2 = conv2(yolo_conv2, 3, [1, 2, 2, 1], 1024, 1024)  # 8*8*1024
    #
    #     yolo_convf3 = conv2(yolo_conv_stride2, 3, [1, 1, 1, 1], 1024, 1024)
    #     yolo_convs3 = conv2(yolo_convf3, 3, [1, 1, 1, 1], 1024, 1024)
    #     tans_1 = tf.transpose(yolo_convs3, [0, 3, 1, 2], name='trans_1')
    #     flat_2 = slim.flatten(tans_1, scope='flat_2')
    #     fc_3 = slim.fully_connected(flat_2, 4096, scope='fc_3')
    #     dropout_4 = slim.dropout(
    #         fc_3, keep_prob=keep_prob, is_training=is_training,
    #         scope='dropout_4')
    #     yolo_output = slim.fully_connected(
    #         dropout_4, yolo_num_outputs, activation_fn=None, scope='fc_5')
    #
    # tf.summary.image('output',tf.transpose(output[0:1,:,:,:],[3,1,2,0]),max_outputs=16)
    # return output, yolo_output

    # add yolo head in the middle of hg_net
    if add_yolo_position == "middle":
        with tf.variable_scope('yolo'):
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    activation_fn=leaky_relu(alpha),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False)
            ):
                # transform 64*64*256 to 56*56*256
                net = slim.conv2d(r3, 256, 5, padding='VALID', scope='conv_64_2_52_1')
                net = slim.conv2d(net, 256, 5, padding='VALID', scope='conv_64_2_52_2')
                # use 3-last layers of yolo_net
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                yolo_output = slim.fully_connected(
                    net, cell_hight * cell_width * ch, activation_fn=None, scope='fc_36')
                yolo_output = tf.reshape(yolo_output,
                                         [-1, ch, cell_hight, cell_width],
                                         name='rs_37')
                yolo_output = tf.transpose(yolo_output, [0, 2, 3, 1], name='trans_38')

            tf.summary.image('output',tf.transpose(output[0:1,:,:,:],[3,1,2,0]),max_outputs=16)
    return output, yolo_output