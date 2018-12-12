import numpy as np

def image_normalization(image):
    #除以255对图像进行归一化
    nor_image=np.asarray(image,dtype=np.float32)
    for i in range(image.shape[0]):
        nor_image[i]=image[i]/255.0
    return nor_image

