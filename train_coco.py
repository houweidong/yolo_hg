import os
import argparse
import datetime
import tensorflow as tf
from utils import config as cfg
from utils.config_utils import update_config
from model.hourglass_yolo_net import HOURGLASSYOLONet
from dataset.coco import Coco
import tensorflow.contrib.slim as slim


class Solver(object):

    def __init__(self, net, data):
        # self.lw = lw
        self.train_mode = cfg.TRAIN_MODE
        self.restore_mode = cfg.RESTORE_MODE
        self.add_yolo_position = cfg.ADD_YOLO_POSITION
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER

        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE

        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        if cfg.OUTPUT_DIR_TASK:
            self.output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.OUTPUT_DIR_TASK)
        else:
            self.output_dir = os.path.join(
                cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # when load model and restore mode is scope, decide which parameters need to exclude(not load)
        # has no use now, because restore mode always is all
        self.cp_ec_scopes = cfg.CHECKPOINT_EXCLUDE_SCOPES
        # when train in scope mode, decide which part parameters train, which part not train
        # not support now, because train mode always is all
        self.train_scopes = cfg.TRAINABLE_SCOPES
        self.save_cfg()

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'hg_yolo')

        self.summary_op = tf.summary.merge_all('train')
        self.summary_op_val = tf.summary.merge_all('val')

        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
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

                    # train
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
                    if step % (self.summary_iter * 1000) == 0:
                        summary_str_val, loss_val, hg_loss_val, yolo_loss_val = self.sess.run(
                            [self.summary_op_val,
                             self.net.loss,
                             self.net.hg_loss,
                             self.net.yolo_loss],
                            feed_dict=val_feed_dict)
                        log_str_val = "VAL   Loss: {:<.3e}  HGLoss: {:<.3e}  " \
                                      "YOLOLoss: {:<.3e}".format(loss_val, hg_loss_val, yolo_loss_val)
                        print(log_str_val)
                    else:
                        summary_str_val, _ = self.sess.run(
                            [self.summary_op_val,
                             self.net.loss],
                            feed_dict=val_feed_dict)

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
        if self.train_mode == "all":
            print("train all parameters")
            variables_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
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
        if self.restore_mode == 'all':
            print('restore all parameters')
            ckpt = tf.train.get_checkpoint_state(self.weights_file)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        # always not use it, because restore_mode always is all
        else:
            print('restore scope parameters')
            slim.assign_from_checkpoint_fn(self.weights_file,
                                           self.get_tuned_variables(),
                                           ignore_missing_vars=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l2', '--l2_regularization', action='store_true', help='use l2 regularization')
    parser.add_argument('-l2f', '--l2_factor', default=0.1, type=float)
    parser.add_argument('-bhm', '--bbox_hm', action='store_true', help='use focal loss')
    parser.add_argument('-bhml', '--bbox_hm_level', default=0, type=int, choices=[i for i in range(8)])
    parser.add_argument('-cs', '--csize', default=64, type=int)
    parser.add_argument('-fc', '--focal_loss', action='store_true', help='use focal loss')
    parser.add_argument('-lw', '--load_weights', action='store_true', help='load weighs from wights dir')
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--position', default="tail", type=str,
                        choices=["tail", "tail_tsp", "tail_conv", "tail_tsp_self",
                                 "tail_conv_deep", "tail_conv_deep_fc", "tail_conv_32", "tail_conv_16"])
    parser.add_argument('--train_mode', default="all", type=str, choices=["all", "scope"])
    parser.add_argument('--restore_mode', default="all", type=str, choices=["all", "scope"])
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    parser.add_argument('--factor', default=0.3, type=float)
    parser.add_argument('--ob_f', default=20.0, type=float)
    parser.add_argument('--noob_f', default=1.0, type=float)
    parser.add_argument('--coo_f', default=100.0, type=float)
    parser.add_argument('--cl_f', default=40.0, type=float)
    args = parser.parse_args()

    update_config(args)
    hg_yolo = HOURGLASSYOLONet('train')
    dataset = Coco()
    solver = Solver(hg_yolo, dataset)

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':
    # python train_coco.py  --gpu 0 --log_dir test
    main()
