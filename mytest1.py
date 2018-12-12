import tensorflow as tf
q = tf.FIFOQueue(1000, tf.float32)
counter = tf.Variable(0.0) # 计数器
increment_op = tf.assign_add(counter, tf.constant(1.0)) # 操作给计数器加一
enquence_op = q.enqueue(counter) # 操作: 让计数器加入队列
# 创建一个队列管理器QueueRunner,用这两个操作相对列q中添加元素,目前我们只使用一个线程
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enquence_op] * 1)
# 启动一个会话,从队列管理器qr中创建线程
# 主线程
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    enquence_threads = qr.create_threads(sess, start=True) # 启用入队线程 # 主线程
    for i in range(5):
        print(sess.run(q.dequeue()))
        # tensorflow.python.framework.errors_impl.CancelledError: Run call was cancelled
