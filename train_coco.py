import os
import argparse
import datetime
import tensorflow as tf
import hg_yolo.config as cfg
from hg_yolo.hourglass_yolo_net import HOURGLASSYOLONet
from dataset.timer import Timer
from dataset.coco import Coco
# from dataset import gene_hm
# from dataset import processing
import tensorflow.contrib.slim as slim
#slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):
        # self.lw = lw
        self.train_op = cfg.TRAIN_OP
        self.add_yolo_position = cfg.ADD_YOLO_POSITION
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.COCO_MAX_ITER

        self.initial_learning_rate = cfg.COCO_LEARNING_RATE
        self.decay_steps = cfg.COCO_DECAY_STEPS
        self.decay_rate = cfg.COCO_DECAY_RATE
        self.staircase = cfg.COCO_STAIRCASE

        self.summary_iter = cfg.COCO_SUMMARY_ITER
        self.save_iter = cfg.COCO_SAVE_ITER
        if cfg.OUTPUT_DIR_TASK:
            self.output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.OUTPUT_DIR_TASK)
        else:
            self.output_dir = os.path.join(
                cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.cp_ec_scopes = cfg.CHECKPOINT_EXCLUDE_SCOPES
        self.train_scopes = cfg.TRAINABLE_SCOPES
        self.save_cfg()

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'hg_yolo')

        self.summary_op = tf.summary.merge_all()
        #self.summary_yolo_op = tf.summary.merge_all('yolo_loss')

        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        #self.optimizer = tf.train.GradientDescentOptimizer(
        #     learning_rate=self.learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        #self.train_op = slim.learning.create_train_op(
        #    self.net.loss, self.optimizer, global_step=self.global_step)\
        self.train_op = self.optimizer.minimize(self.net.loss,
                                                self.global_step,
                                                self.get_trainable_variables())

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
        self.data.sess = self.sess

        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            # self.saver.restore(self.sess, self.weights_file)
            slim.assign_from_checkpoint_fn(self.weights_file,
                                           self.get_tuned_variables(),
                                           ignore_missing_vars=True)

        self.writer.add_graph(self.sess.graph)

    def train(self):
        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):

            load_timer.tic()
            images, labels_det, labels_kp = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels_det: labels_det,
                         self.net.labels_kp: labels_kp}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:

                    train_timer.tic()
                    summary_str, loss, hg_loss, yolo_loss, _ = self.sess.run(
                        [self.summary_op,
                         self.net.loss,
                         self.net.hg_loss,
                         self.net.yolo_loss,
                         self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    # log_str = "Loss head: {} Epoch: {}, Step: {}, Learning rate: {}, " \
                    #           "Loss: {:5.3f}\nSpeed: {:.3f}s/iter, " \
                    #           "Load: {:.3f}s/iter, Remain: {}".format(
                    log_str = "Epoch: {}  Step: {:<5}  Learning rate:  {:.3e}  " \
                              "Loss: {:<.3e}  HGLoss: {:<.3e}  YOLOLoss: {:<.3e}  Remain: {}".format(
                                step // self.data.coco_epoch_size + 1,
                                int(step),
                                self.learning_rate.eval(session=self.sess),
                                loss,
                                hg_loss,
                                yolo_loss,
                                #train_timer.average_time,
                                #load_timer.average_time,
                                train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
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
            print("save cfg to", os.path.join(self.output_dir, 'config.txt'))
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

    def get_tuned_variables(self):
        # TODO RETURN TUNED VARIABLES LIST
        variables_to_restore = []
        for var in slim.get_model_variables():
            exclued = False
            for exclusion in self.cp_ec_scopes:
                if var.op.name.startswith(exclusion):
                    exclued = True
                    break
            if not exclued:
                variables_to_restore.append(var)
        return variables_to_restore

    def get_trainable_variables(self):
        variables_to_train = []
        if self.train_op == "all":
            variables_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            print("TRAIN ALL VAR")
        elif self.train_op == "sp" and self.add_yolo_position == "middle":
            print("TRAIN SCOPE VAR")
            for sp in self.train_scopes:
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, sp)
                variables_to_train.extend(variables)
        # TODO do not support now
        else:
            print('if add yolo to the tail of hg_net, can not load weight from YOlo_small.ckpt!')
            raise RuntimeError('Input Error')
        # delete duplicated variables
        return list(set(variables_to_train))

# def update_config_paths(data_dir, weights_file):
#     cfg.DATA_PATH = data_dir
#     cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
#     cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
#     cfg.OUTPUT_DIR = os.path.join(data_dir, 'output')
#     cfg.WEIGHTS_DIR = os.path.join(data_dir, 'weights')
#
#     cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)
def update_config_paths(weights_file):
    #cfg.DATA_PATH = data_dir
    #cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    #cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    #cfg.OUTPUT_DIR = 'output'
    #cfg.WEIGHTS_DIR = 'weights'
    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lw', '--load_weights', action='store_true', help='load weighs from wights dir')
    parser.add_argument('--position', default="tail", type=str, choices=["tail", "middle"])
    parser.add_argument('--train_op', default="all", type=str, choices=["all", "sp"])
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', type=str)
    #parser.add_argument('--option', default=1, type=int, help='decide where the dataset from')
    args = parser.parse_args()

    # if args.data_dir != cfg.DATA_PATH:
    #     update_config_paths(args.data_dir, args.weights)
    if args.load_weights:
        # update_config_paths(args.data_dir, args.weights)
        update_config_paths(args.weights)
        cfg.TRAIN_OP = args.train_op
    else:
        cfg.TRAIN_OP = "all"
    if args.gpu is not None:
        cfg.GPU = args.gpu
    if args.log_dir:
        cfg.OUTPUT_DIR_TASK = args.log_dir
    cfg.ADD_YOLO_POSITION = args.position
    print("YOLO POSITION: {}".format(cfg.ADD_YOLO_POSITION))
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    hg_yolo = HOURGLASSYOLONet()
    dataset = Coco('train')

    solver = Solver(hg_yolo, dataset)

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
