import tensorflow as tf
import numpy as np

# a = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
#      [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]
# b = np.reshape(a, (-1))
# c = np.reshape(b, (4, 2, 3))
# print(c)
# c = np.transpose(c, (1, 2, 0))
# print(c)
# print(c[:, :, 0])
a = [1]
b = a * 5
print(a, b)
a = tf.constant([0.0])
print(a)
