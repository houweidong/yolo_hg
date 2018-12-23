import os
import argparse
import datetime
import tensorflow as tf
import utils.config as cfg
from model.hourglass_yolo_net import HOURGLASSYOLONet
from utils.timer import Timer
from dataset.hgvoc import hg_voc
# from dataset import gene_hm
# from dataset import processing

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER

        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE

        self.initial_learning_rate_hg = cfg.LEARNING_RATE_HG
        self.decay_steps_hg = cfg.DECAY_STEPS_HG
        self.decay_rate_hg = cfg.DECAY_RATE_HG
        self.staircase_hg = cfg.STAIRCASE_HG

        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'hg_yolo')

        self.summary_hg_op = tf.summary.merge_all('hg_loss')
        self.summary_yolo_op = tf.summary.merge_all('yolo_loss')

        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        #self.optimizer = tf.train.GradientDescentOptimizer(
        #     learning_rate=self.learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        self.learning_rate_hg = tf.train.exponential_decay(
            self.initial_learning_rate_hg, self.global_step, self.decay_steps_hg,
            self.decay_rate_hg, self.staircase_hg, name='learning_rate_hg')
        # self.optimizer_hg = tf.train.GradientDescentOptimizer(
        #    learning_rate=self.learning_rate_hg)
        self.optimizer_hg = tf.train.RMSPropOptimizer(self.learning_rate_hg)

        self.train_hg_op = slim.learning.create_train_op(
            self.net.hg_loss, self.optimizer_hg, global_step=self.global_step)
        self.train_yolo_op = slim.learning.create_train_op(
            self.net.yolo_loss, self.optimizer, global_step=self.global_step)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
        self.data.sess = self.sess

        self.cp_ec_scopes = cfg.CHECKPOINT_EXCLUDE_SCOPES
        self.train_scopes = cfg.TRAINABLE_SCOPES
        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            # self.saver.restore(self.sess, self.weights_file)
            slim.assign_from_checkpoint_fn(self.sess,
                                           self.weights_file,
                                           self.get_tuned_variables(),
                                           ignore_missing_vars=True)

        self.writer.add_graph(self.sess.graph)

    def train(self):
        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):

            load_timer.tic()
            images, labels, sign = self.data.all_get()
            train_op = self.train_yolo_op
            summary_op = self.summary_yolo_op
            loss_op = self.net.yolo_loss
            learning_rate =self.learning_rate
            if sign == "hourglass":
                train_op = self.train_hg_op
                summary_op = self.summary_hg_op
                loss_op = self.net.hg_loss
                learning_rate = self.learning_rate_hg
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            if step % self.summary_iter == 0:
                if step % self.summary_iter == 0:

                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [summary_op, loss_op, train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    # log_str = "Loss head: {} Epoch: {}, Step: {}, Learning rate: {}, " \
                    #           "Loss: {:5.3f}\nSpeed: {:.3f}s/iter, " \
                    #           "Load: {:.3f}s/iter, Remain: {}".format(
                    log_str = "Loss head: {:<9}  Epoch: {}  Step: {:<5}  Learning rate:  {:.3e}  " \
                              "Loss: {:<.3e}  Remain: {}".format(
                                sign,
                                self.data.epoch,
                                int(step),
                                learning_rate.eval(session=self.sess),
                                loss,
                                #train_timer.average_time,
                                #load_timer.average_time,
                                train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [summary_op, train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                self.sess.run(train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)
        self.coord.request_stop()
        self.coord.join(self.threads)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(data_dir, 'output')
    cfg.WEIGHTS_DIR = os.path.join(data_dir, 'weights')

    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lw', '--load_weights', action='store_true', help='load weighs from wights dir')
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--option', default=1, type=int, help='decide where the dataset from')
    args = parser.parse_args()

    # if args.data_dir != cfg.DATA_PATH:
    #     update_config_paths(args.data_dir, args.weights)
    if args.load_weights:
        update_config_paths(args.data_dir, args.weights)

    if args.gpu is not None:
        cfg.GPU = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    hg_yolo = HOURGLASSYOLONet()
    dataset = hg_voc('train', option=args.option)

    solver = Solver(hg_yolo, dataset)

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
