# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 9:49
# @Author  : weic
# @FileName: view_result.py
# @Software: PyCharm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from dataset import gene_hm

def tiny_adjust(label):
    """

    :param label: 一行预测结果，x,y,pro *16个
    :return:
    """
    re_label=np.zeros_like(label,np.float32)
    for i in range(len(label)):
        re_label[i]=label[i]
        if (label[i][2] > 0.98):

            # print(label[i][2])
            or_w = int(label[i][0])
            or_h = int(label[i][1])
            print(or_w, or_h)
            a = np.reshape(np.linspace(or_w - 0.49, or_w + 0.5, 20), (-1, 1))
            b = np.reshape(np.linspace(or_h - 1, or_h + 1, 20), (-1, 1))
            vit_label = np.expand_dims(np.concatenate([a, b], axis=1), axis=0)
            hm = gene_hm.batch_genehm(1, vit_label)[0]
            hm_result = []
            for j in range(20):
                hm_result.append([vit_label[0][j][0], vit_label[0][j][1], np.max(hm[j])])

            ad_result = np.asarray(hm_result).reshape(-1, 3)
            # print(label[i][2],)
            ad_result[:, 2] = np.abs(ad_result[:, 2] - label[i][2])
            # print(ad_result)
            # print(np.min(ad_result[:,2]))
            index = np.argmin(ad_result[:, 2], axis=0)
            re_w = ad_result[index, 0]
            re_h = ad_result[index, 1]
            re_label[i, 0] = re_w
            re_label[i, 1] = re_h
            # if not np.array_equal(re_label,label):
            #     print(re_label)

    return re_label




with open('../xiaolunwen/result_diff.txt') as f:
    results=f.readlines()

    for result in results:
        #print(type(result))
        #print(result)
        list_result=result.split(' ')
        #print(list_result)
        # img_name=list_result[0].split('/')[-1]
        #img_name=list_result[0].split('\\')[-1]
        img_path = list_result[0]
        #imgPath=os.path.join('../test',img_name)
        #print(list_result[1].split(' '))
        label=np.asarray([float(x) for x in list_result[1:-1]]).reshape(-1,3)




        try:
            img = Image.open(img_path)
        except Exception as info:
            # print(info)
            continue
        else:
            width, height = img.size

            plt.imshow(img)
            # plt.plot(label[0:7,0], label[0:7,1], 'r+')
            # plt.plot(label[9:13,0], label[9:13,1], 'r+')
            # plt.plot(label[15,0], label[15,1], 'r+')
            # plt.scatter(label[0:7,0], label[0:7,1],c='r',marker='+')
            # plt.scatter(label[9:13,0], label[9:13,1],c='r',marker='+')
            # plt.scatter(label[15,0], label[15,1],c='r',marker='+')

            plt.scatter(label[:,0],label[:,1],c='r',marker='+')

            # plt.plot(ta_label[:,0] * width / 64, ta_label[:,1] * height / 64, 'g+')

            # plt.plot(x[15],y[15],'g+')
            plt.show()
