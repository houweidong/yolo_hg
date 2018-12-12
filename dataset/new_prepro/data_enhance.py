import tensorflow as tf
import numpy as np
import random


def _gen_full_boundingBox(label,width,height):

    re_width=tf.cast(width,tf.int32)
    re_height=tf.cast(height,tf.int32)
    re_label=tf.reshape(label,(-1,3))
    l_min=tf.reduce_min(re_label,axis=0)
    l_max=tf.reduce_max(re_label,axis=0)

    left_margin=tf.cast(tf.floor(l_min[0]),tf.int32)
    top_margin=tf.cast(tf.floor(l_min[1]),tf.int32)
    right_margin=tf.cast(tf.floor(l_max[0]),tf.int32)
    bottom_margin=tf.cast(tf.floor(l_max[1]),tf.int32)

    # left_margin=l_min[0]
    # top_margin=l_min[1]
    # right_margin=l_max[0]
    # bottom_margin=l_max[1]

    left0=tf.random_uniform([1],0,left_margin+1,dtype=tf.int32)
    left=tf.random_uniform([1],0,left0[0]+1,dtype=tf.int32)
    #left=tf.random_uniform([1],0,left1[0]+1,dtype=tf.int32)

    top0=tf.random_uniform([1],0,top_margin+1,dtype=tf.int32)
    top=tf.random_uniform([1],0,top0[0]+1,dtype=tf.int32)
    # top=tf.random_uniform([1],0,top1[0]+1,dtype=tf.int32)

    right0=tf.random_uniform([1],right_margin,re_width,dtype=tf.int32)
    right=tf.random_uniform([1],right0[0],re_width,dtype=tf.int32)
    # right=tf.random_uniform([1],right1[0],re_width,dtype=tf.int32)


    bottom0=tf.random_uniform([1],bottom_margin,re_height,dtype=tf.int32)
    bottom=tf.random_uniform([1],bottom0[0],re_height,dtype=tf.int32)
    # bottom=tf.random_uniform([1],bottom1[0],re_height,dtype=tf.int32)
    new_width=right-left
    new_height=bottom-top

    return (top[0],left[0],new_height[0],new_width[0])


def _relabel_ac_bbox(label,bbox):
    re_label=tf.reshape(label,(-1,3))
    top=tf.cast(bbox[0],tf.float32)
    left=tf.cast(bbox[1],tf.float32)
    x=tf.reshape(re_label[:,0]-left,(-1,1))
    y=tf.reshape(re_label[:,1]-top,(-1,1))
    result=tf.concat([x,y],axis=1)

    return result


def _re_gtbbox_ac_bbox(gt_bbox,bbox):
    re_label = tf.reshape(gt_bbox, (-1, 2))
    top = tf.cast(bbox[0], tf.int64)
    left = tf.cast(bbox[1], tf.int64)
    x = tf.reshape(re_label[:, 0] - left, (-1, 1))
    y = tf.reshape(re_label[:, 1] - top, (-1, 1))
    result = tf.concat([x, y], axis=1)
    return result


def random_crop_img(img,label,width,height):
    ":return:crop_img,bbox:(top[0],left[0],new_height[0],new_width[0])"
    bbox=_gen_full_boundingBox(label,width,height)
    crop_img=tf.image.crop_to_bounding_box(img,bbox[0],bbox[1],bbox[2],bbox[3])
    return crop_img, bbox


def random_crop_img_bbox(img, label, width, height, gt_bbox):
    bbox = _gen_full_boundingBox(label, width, height)
    crop_img = tf.image.crop_to_bounding_box(img, bbox[0], bbox[1], bbox[2], bbox[3])
    re_label = _relabel_ac_bbox(label, bbox)
    re_gt_bbox = _re_gtbbox_ac_bbox(gt_bbox,bbox)
    return crop_img, re_label, re_gt_bbox, bbox[2], bbox[3]


def adjust_image(color_ordering,image,fast=False):
    if fast:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)

      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)

      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)

      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return image


def mirror_img(img,label,img_width,has_box=False,new_gt_bbox=None):
    #label (-1,2)
    flip_img=tf.image.flip_left_right(img)
    re_width = tf.cast(img_width, tf.float32)
    x = tf.reshape(re_width-label[:, 0] , (-1, 1))
    y = tf.reshape(label[:, 1], (-1, 1))
    new_x=[]
    new_y=[]
    #print(label,x,y)
    for n in range(34):
        i=n%17
        if i == 0 :
            l_x = x[i, 0]
            l_y= y[i,0]
        else:
            if i % 2 == 0:
                l_x= x[i - 1, 0]
                l_y = y[i-1, 0]

            else:
                l_x= x[i + 1][0]
                l_y = y[i+1, 0]

        new_x.append(tf.reshape(l_x,(-1,1)))
        new_y.append(tf.reshape(l_y,(-1,1)))
    t_new_x=tf.concat(new_x,axis=0)
    t_new_y=tf.concat(new_y,axis=0)
    flip_label= tf.concat([t_new_x, t_new_y], axis=1)
    if has_box:
        new_gt_bbox = _mirror_gt_bbox(new_gt_bbox, re_width)
    else:
        new_gt_bbox = np.zeros([2,2],dtype=np.int64)
    return flip_img,flip_label,new_gt_bbox


def _mirror_gt_bbox(gt_bbox, img_width):
    re_width = tf.cast(img_width, tf.int64)
    x = tf.reshape(re_width - gt_bbox[:, 0], (-1, 1))
    y = tf.reshape(gt_bbox[:, 1], (-1, 1))
    new_x = []

    for i in range(4):
        if i<2:
            l_x= x[(i+1)%2,0]
        else:
            l_x= x[(i+1)%2+2,0]
        new_x.append(tf.reshape(l_x, (-1, 1)))
    t_new_x = tf.concat(new_x, axis=0)
    flip_bbox = tf.concat([t_new_x, y], axis=1)
    return flip_bbox


def do_not_mirror(adj_image, re_label, has_bbox, gt_bbox):
    if has_bbox:
        bbox=gt_bbox
    else:
        bbox=np.zeros([2,2],dtype=np.int64)
    return adj_image,re_label,bbox



def do_enhance(image, label, width, height, has_box = False ,gt_bbox=None):
    """
    :param image:
    :param label:
    :param width:
    :param height:
    :return: label shape is (nPoints,2)
    """

    rand_flip_num = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.math.less(rand_flip_num, 0.5)
    adj_image = adjust_image(1, image)
    adj_image, bbox = random_crop_img(adj_image, label, width, height)
    re_width = bbox[3]
    re_height = bbox[2]
    re_label = _relabel_ac_bbox(label, bbox)
    if has_box:# has bbox
        new_gt_bbox=_re_gtbbox_ac_bbox(gt_bbox,bbox)
        mirror_result=tf.cond(mirror_cond,
                              lambda : mirror_img(adj_image, re_label, re_width, has_box, new_gt_bbox),
                              lambda : do_not_mirror(adj_image, re_label, has_box, new_gt_bbox))

        return mirror_result, re_width, re_height
    else:
        mirror_result = tf.cond(mirror_cond,
                                lambda :mirror_img(adj_image, re_label, re_width),
                                lambda :do_not_mirror(adj_image, re_label,has_box,new_gt_bbox))
        return mirror_result,  re_width, re_height

