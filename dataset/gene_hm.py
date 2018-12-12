import numpy as np
import math
HM_HEIGHT=64
HM_WIDTH=64
nPoints=16
def resize_label(label):
    return label*64/256.
def _makeGaussian(height, width, sigma, center):
    """
    以center为中心生成值逐渐减小的矩阵，中心值为一
    :param height:
    :param width:
    :param sigma:
    :param center:
    :return: 一个height和width的矩阵
    """
    x = np.arange(0, width, 1, np.float32)
    y = np.arange(0, height, 1, np.float32)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2.0) * ((x-x0)**2 + (y-y0)**2) / sigma**2)
    #return np.exp(-4 * np.log(2.0) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def one_point_hm(height, width ,joints):
    num_joints = joints.shape[0]
    r_hm = np.zeros((num_joints, height, width), dtype=np.float32)
    for i in range(num_joints):
        r_hm[i] = np.zeros((height, width), dtype=np.float32)
        if np.array_equal(joints[i], [0, 0]) or np.array_equal(joints[i], [256., 0]):
            continue
        else:
            x = math.floor(joints[i][0]) if joints[i][0]-int(joints[i][0])<0.5 else math.ceil(joints[i][0])
            y = math.floor(joints[i][1]) if joints[i][1]-int(joints[i][1])<0.5 else math.ceil(joints[i][1])
            r_hm[i,x,y] = 1
    return r_hm


def generate_hm(height, width ,joints, maxlenght, num_joints=0):
    #print(joints)
    num_joints = joints.shape[0] if num_joints==0 else num_joints
    r_hm = np.zeros((num_joints,height, width), dtype=np.float32)
    for i in range(num_joints):
        if np.array_equal(joints[i],[0,0]) or np.array_equal(joints[i], [64., 0]) :
            r_hm[i] = np.zeros((height, width), dtype=np.float32)
        elif np.array_equal(joints[i],[HM_WIDTH,0]):
            r_hm[i] = np.zeros((height, width), dtype=np.float32)
        else:
            # if not (np.array_equal(joints[i], [0, 0])) :#and weight[i] == 1:
            s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
            r_hm[i]= _makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
    return r_hm


def batch_genehm(batch_size,l,if_gauss):
    re_label=resize_label(l)
    label = np.zeros((batch_size, nPoints, HM_HEIGHT,HM_WIDTH), dtype=np.float32)
    if if_gauss:
        for i in range(batch_size):
            label[i] =generate_hm(HM_HEIGHT,HM_WIDTH, re_label[i], 256)
    else:

        for i in range(batch_size):
            label[i] =one_point_hm(HM_HEIGHT,HM_WIDTH, re_label[i])
    return label


def batch_genehm_for_coco(batch_size,l,num_points,if_gauss=True):
    re_label = resize_label(l)
    label = np.zeros((batch_size, num_points, HM_HEIGHT, HM_WIDTH), dtype=np.float32)
    if if_gauss:
        for i in range(batch_size):
            label[i] = generate_hm(HM_HEIGHT, HM_WIDTH, re_label[i][0:num_points], 64, num_points)+generate_hm(HM_HEIGHT, HM_WIDTH, re_label[i][num_points:], 64,num_points)
    else:
        for i in range(batch_size):
            label[i] = one_point_hm(HM_HEIGHT, HM_WIDTH, re_label[i])+one_point_hm(HM_HEIGHT, HM_WIDTH, re_label[i+nPoints])
    return label


# joints=np.array([[[1,1],[15.6,15]]])
# hm=batch_genehm(1,joints,False)
# # hm=one_point_hm(64,64,joints)
# print(hm.shape)
# print(hm[0,1,14:20,14:20])