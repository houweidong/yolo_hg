import numpy as np
import random
import tensorflow as tf

q = tf.FIFOQueue(1000, tf.float32)
counter = tf.Variable(0.0)  # 计数器
increment_op = tf.assign_add(counter, tf.constant(1.0))  # 操作给计数器加一
enquence_op = q.enqueue(counter)  # 操作: 让计数器加入队列
# 第一种情况,在关闭其他线程之后(除主线程之外的其它线程),调用出队操作
print('第一种情况,在关闭其他线程之后(除主线程之外的其它线程),调用出队操作')
# 创建一个队列管理器QueueRunner,用这两个操作相对列q中添加元素,目前我们只使用一个线程
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enquence_op] * 1)
# 主线程
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Coordinator: 协调器, 协调线程间的关系,可以视为一种信号量,用来做同步
coord = tf.train.Coordinator()
# 启动入队线程,协调器是线程的参数
enqueue_threads = qr.create_threads(sess,coord=coord,start=True)
# 主线程
for i in range(0,10):
    print(sess.run(q.dequeue()))
coord.request_stop()  # 通知其他线程关闭
coord.join(enqueue_threads)  # join 操作等待其他线程结束,其他所有线程关闭之后,这一函数才能返回
print('第二种情况: 在队列线程关闭之后,调用出队操作-->处理tf.errors.OutOfRange错误')
# q启动入队线程
enqueue_threads = qr.create_threads(sess,coord=coord,start=True) # 主线程 coord.request_stop() # 通知其他线程关闭 for j in range(0,10): try: print(sess.run(q.dequeue())) except tf.errors.OutOfRangeError: break coord.join(enqueue_threads) # join 操作等待其他线程结束,其他所有线程关闭之后,这一函数才能返回
