import os
import argparse
import datetime
import tensorflow as tf
from utils import config as cfg
from model.hourglass_yolo_net import HOURGLASSYOLONet
from dataset.coco import Coco
import tensorflow.contrib.slim as slim


class Solver(object):

    def __init__(self, net, data):
        # self.lw = lw
        self.train_sp = cfg.TRAIN_OP
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

        self.summary_op = tf.summary.merge_all('train')
        self.summary_op_val = tf.summary.merge_all('val')
        # self.summary_yolo_op = tf.summary.merge_all('yolo_loss')

        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        # self.optimizer = tf.train.GradientDescentOptimizer(
        #      learning_rate=self.learning_rate)
        # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        # self.train_op = slim.learning.create_train_op(
        #    self.net.loss, self.optimizer, global_step=self.global_step)\
        self.train_op = self.optimizer.minimize(self.net.loss,
                                                self.global_step,
                                                self.get_trainable_variables())
        self.val_op = self.net.loss

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
        self.data.sess = self.sess

        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            self.restore()

        self.writer.add_graph(self.sess.graph)

    def train(self):

        step = 1
        while True:
            # for step in range(1, self.max_iter + 1):
            images, labels_det, labels_kp = self.data.get("train")
            feed_dict = {self.net.images: images,
                         self.net.labels_det: labels_det,
                         self.net.labels_kp: labels_kp}

            if step % self.summary_iter == 0:

                val_im_bt, val_det_bt, val_kp_bt = self.data.get("val")
                val_feed_dict = {self.net.images: val_im_bt,
                                 self.net.labels_det: val_det_bt,
                                 self.net.labels_kp: val_kp_bt}

                if step % (self.summary_iter * 10) == 0:

                    summary_str, loss, hg_loss, yolo_loss, _ = \
                        self.sess.run([self.summary_op, self.net.loss,
                                       self.net.hg_loss, self.net.yolo_loss, self.train_op],
                                      feed_dict=feed_dict)

                    log_str = "TRAIN Loss: {:<.3e}  HGLoss: {:<.3e}  YOLOLoss: {:<.3e}  " \
                              "Epoch: {}  Step: {:<5}  Learning rate:  {:.3e}" \
                        .format(
                                loss,
                                hg_loss,
                                yolo_loss,
                                step // cfg.COCO_EPOCH_SIZE + 1,
                                int(step),
                                self.learning_rate.eval(session=self.sess))
                    print(log_str)

                    # val
                    summary_str_val, loss_val, hg_loss_val, yolo_loss_val = self.sess.run(
                        [self.summary_op_val,
                         self.net.loss,
                         self.net.hg_loss,
                         self.net.yolo_loss],
                        feed_dict=val_feed_dict)
                    log_str_val = "VAL   Loss: {:<.3e}  HGLoss: {:<.3e}  " \
                                  "YOLOLoss: {:<.3e}".format(loss_val, hg_loss_val, yolo_loss_val)
                    print(log_str_val)

                    # caculate AP for all val set
                    # if step % (self.summary_iter * 10) == 0:
                    #     print("AP: ", self.evaluate())

                else:
                    # train
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)

                    # val
                    summary_str_val, _ = self.sess.run(
                        [self.summary_op_val,
                         self.net.loss],
                        feed_dict=val_feed_dict)
                self.writer.add_summary(summary_str, step)
                self.writer.add_summary(summary_str_val, step)

            else:
                self.sess.run(self.train_op, feed_dict=feed_dict)

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)
            step += 1
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
        if self.train_sp == "all":
            variables_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            print("TRAIN ALL VAR")
        # elif self.train_sp == "sp" and self.add_yolo_position == "middle":
        #     print("TRAIN SCOPE VAR")
        #     for sp in self.train_scopes:
        #         variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, sp)
        #         variables_to_train.extend(variables)
        # else:
        #     print('if add yolo to the tail of hg_net, can not load weight from YOlo_small.ckpt!')
        #     raise RuntimeError('Input Error')
        # delete duplicated variables
        else:
            print('not support now')
        return list(set(variables_to_train))

    def restore(self):
        if self.train_sp == 'all':
            ckpt = tf.train.get_checkpoint_state(self.weights_file)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            slim.assign_from_checkpoint_fn(self.weights_file,
                                           self.get_tuned_variables(),
                                           ignore_missing_vars=True)


# def update_config_paths(data_dir, weights_file):
#     cfg.DATA_PATH = data_dir
#     cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
#     cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
#     cfg.OUTPUT_DIR = os.path.join(data_dir, 'output')
#     cfg.WEIGHTS_DIR = os.path.join(data_dir, 'weights')
#
#     cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)
# def update_config_paths(weights_file):
#     #cfg.DATA_PATH = data_dir
#     #cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
#     #cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
#     #cfg.OUTPUT_DIR = 'output'
#     #cfg.WEIGHTS_DIR = 'weights'
#     cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def update_config(args):
    if args.gpu is not None:
        cfg.GPU = args.gpu
    if args.log_dir:
        cfg.OUTPUT_DIR_TASK = args.log_dir

    cfg.ADD_YOLO_POSITION = args.position
    if args.load_weights:
        # update_config_paths(args.data_dir, args.weights)
        cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, args.weights)
    cfg.LOSS_FACTOR = args.factor
    cfg.OBJECT_SCALE = args.ob_f
    cfg.NOOBJECT_SCALE = args.noob_f
    cfg.COORD_SCALE = args.coo_f
    cfg.CLASS_SCALE = args.cl_f
    cfg.CELL_SIZE = args.csize

    print("YOLO POSITION: {}".format(cfg.ADD_YOLO_POSITION))
    print("LOSS_FACTOR:{}  OB_SC:{}  "
          "NOOB_SC:{}  COO_SC:{}".format(args.factor, args.ob_f, args.noob_f, args.coo_f))
    print("LR: {}".format(cfg.COCO_LEARNING_RATE))
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lw', '--load_weights', action='store_true', help='load weighs from wights dir')
    parser.add_argument('--position', default="tail", type=str,
                        choices=["tail", "tail_tsp", "tail_conv", "tail_tsp_self",
                                 "tail_conv_deep", "tail_conv_deep_fc"])
    parser.add_argument('--train_op', default="all", type=str, choices=["all", "sp"])
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--gpu', type=str)
    # parser.add_argument('--factor', default=0.05, type=float)
    parser.add_argument('--ob_f', default=1.0, type=float)
    parser.add_argument('--noob_f', default=1.0, type=float)
    parser.add_argument('--coo_f', default=5.0, type=float)
    parser.add_argument('--cl_f', default=2.0, type=float)
    parser.add_argument('--csize', default=64, type=int)
    args = parser.parse_args()

    update_config(args)
    hg_yolo = HOURGLASSYOLONet('train')
    dataset = Coco()
    solver = Solver(hg_yolo, dataset)

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':
    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
