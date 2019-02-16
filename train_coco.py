import os
import argparse
import datetime
import tensorflow as tf
from utils import config as cfg
from utils.config_utils import update_config
from model.hourglass_yolo_net_multi_gpu import HOURGLASSYOLONet
from dataset.coco import Coco
import tensorflow.contrib.slim as slim


class Solver(object):

    def __init__(self, net, data):
        # self.lw = lw
        self.gpu_number = cfg.GPU_NUMBER
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

        self.ckpt_file = os.path.join(self.output_dir, 'hg_yolo')
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        # self.labels_det, self.labels_kp = self.define_holder_det_kp()
        # self.images = self.net.images

    def define_holder_det_kp(self):

        labels_det = tf.placeholder(
            tf.float32,
            [self.net.batch_size * self.gpu_number, self.net.cell_size, self.net.cell_size,
             self.net.num_class + 5 if self.net.num_class != 1 else 5])
        labels_kp = tf.placeholder(
            tf.float32,
            [self.net.batch_size * self.gpu_number, self.net.nPoints, self.net.hg_cell_size,
             self.net.hg_cell_size])
        return labels_det, labels_kp

    def train(self):

        with tf.device("/cpu:0"):
            # self.train_op = self.optimizer.minimize(self.net.loss,
            #                                         self.global_step,
            #                                         self.get_trainable_variables())
            # global_step = tf.train.get_or_create_global_step()
            tower_grads = []
            tower_loss = []
            tower_loss_board = []

            labels_det_hd, labels_kp_hd = self.define_holder_det_kp()
            images_hd = self.net.images
            opt = tf.train.RMSPropOptimizer(self.learning_rate)
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.gpu_number):
                    with tf.device("/gpu:%d" % i):
                        with tf.name_scope("tower_%d" % i) as scope:
                            x = images_hd[i * self.net.batch_size:(i + 1) * self.net.batch_size]
                            y_det = labels_det_hd[i * self.net.batch_size:(i + 1) * self.net.batch_size]
                            y_kp = labels_kp_hd[i * self.net.batch_size:(i + 1) * self.net.batch_size]
                            hg_logits, yolo_logits = self.net.build_network(x)
                            tf.get_variable_scope().reuse_variables()
                            loss, hg_loss, yolo_loss, loss_board = \
                                self.net.loss_layer([hg_logits, yolo_logits],
                                                    [y_det, y_kp], scope)
                            tower_loss.append((loss, hg_loss, yolo_loss))
                            tower_loss_board.append(loss_board)
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)
            grads = self.average_gradients(tower_grads)
            loss_mean, hg_loss_mean, yolo_loss_mean = self.average_loss(tower_loss)
            summary_op, summary_op_val = self.define_summary_op(tower_loss_board)
            train_op = opt.apply_gradients(grads, self.global_step)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            self.data.sess = sess

            if self.weights_file is not None:
                print('Restoring weights from: ' + self.weights_file)
                self.restore(sess, saver)

            self.writer.add_graph(sess.graph)
            step = 1
            while step < self.max_iter:
                # for step in range(1, self.max_iter + 1):
                images, labels_det, labels_kp = self.data.get("train")
                feed_dict = {images_hd: images,
                             labels_det_hd: labels_det,
                             labels_kp_hd: labels_kp}

                if step % self.summary_iter == 0:

                    val_im_bt, val_det_bt, val_kp_bt = self.data.get("val")
                    val_feed_dict = {images_hd: val_im_bt,
                                     labels_det_hd: val_det_bt,
                                     labels_kp_hd: val_kp_bt}

                    if step % (self.summary_iter * 10) == 0:

                        # train
                        summary_str, loss_rs, hg_loss_rs, yolo_loss_rs, _ = \
                            sess.run([summary_op, loss_mean,
                                      hg_loss_mean, yolo_loss_mean, train_op],
                                     feed_dict=feed_dict)

                        log_str = "TRAIN Loss: {:<.3e}  HGLoss: {:<.3e}  YOLOLoss: {:<.3e}  " \
                                  "Epoch: {}  Step: {:<5}  Learning rate:  {:.3e}" \
                            .format(loss_rs,
                                    hg_loss_rs,
                                    yolo_loss_rs,
                                    step // cfg.COCO_EPOCH_SIZE + 1,
                                    int(step),
                                    self.learning_rate.eval(session=sess))
                        print(log_str)

                        # val
                        if step % (self.summary_iter * 1000) == 0:
                            summary_str_val, loss_val, hg_loss_val, yolo_loss_val = sess.run(
                                [summary_op_val,
                                 loss_mean,
                                 hg_loss_mean,
                                 yolo_loss_mean],
                                feed_dict=val_feed_dict)
                            log_str_val = "VAL   Loss: {:<.3e}  HGLoss: {:<.3e}  " \
                                          "YOLOLoss: {:<.3e}".format(loss_val, hg_loss_val, yolo_loss_val)
                            print(log_str_val)
                        else:
                            summary_str_val, _ = sess.run([summary_op_val, loss_mean], feed_dict=val_feed_dict)

                        # caculate AP for all val set
                        # if step % (self.summary_iter * 10) == 0:
                        #     print("AP: ", self.evaluate())

                    else:
                        # train
                        summary_str, _ = sess.run([summary_op, train_op], feed_dict=feed_dict)
                        # val
                        summary_str_val, _ = sess.run([summary_op_val, loss_mean], feed_dict=val_feed_dict)
                    self.writer.add_summary(summary_str, step)
                    self.writer.add_summary(summary_str_val, step)

                else:
                    sess.run(train_op, feed_dict=feed_dict)

                if step % self.save_iter == 0:
                    print('{} Saving checkpoint file to: {}'.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.output_dir))
                    saver.save(
                        sess, self.ckpt_file, global_step=self.global_step)
                step += 1
            coord.request_stop()
            coord.join(threads)

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

    def restore(self, sess, saver):
        if self.restore_mode == 'all':
            print('restore all parameters')
            ckpt = tf.train.get_checkpoint_state(self.weights_file)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
        # always not use it, because restore_mode always is all
        else:
            print('restore scope parameters')
            slim.assign_from_checkpoint_fn(self.weights_file,
                                           self.get_tuned_variables(),
                                           ignore_missing_vars=True)

    def define_summary_op(self, tower_loss_board):
        average_loss_board = Solver.average_loss(tower_loss_board)
        name_lt = ['train', 'val']
        loss_name = ['/yolo/object_loss', '/yolo/noobject_loss', '/yolo/coord_loss',
                     '/yolo/yolo_loss', '/hg_loss', '/total_loss']
        if self.net.num_class != 1:
            loss_name.insert(0, '/yolo/class_loss')
        for name in name_lt:
            for i, l in enumerate(average_loss_board):
                tf.summary.scalar(name + loss_name[i], l, collections=[name])
        # have to define this after the net define which has the summary scalar define
        summary_op = tf.summary.merge_all('train')
        summary_op_val = tf.summary.merge_all('val')
        return summary_op, summary_op_val

    @staticmethod
    def average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    @staticmethod
    def average_loss(tower_loss):
        average_loss = []
        for loss in zip(*tower_loss):
            losses = []
            for l in loss:
                expend_l = tf.expand_dims(l, 0)
                losses.append(expend_l)
            losses_contact = tf.concat(losses, 0)
            losses_mean = tf.reduce_mean(losses_contact, 0)
            average_loss.append(losses_mean)
        return average_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', default=7, type=int)
    parser.add_argument('-hhmdl', '--hg_hm_diff_level', default=0, type=int)
    parser.add_argument('-hhml', '--hg_hm_level', default=1, type=int)
    parser.add_argument('-na', '--number_anchors', default=7, type=int, choices=[7, 10, 13])
    parser.add_argument('-yv', '--yolo_version', default='1', type=str)
    parser.add_argument('-md', '--number_models', default=1, type=int)
    parser.add_argument('-ns', '--number_stacks', default=3, type=int)
    parser.add_argument('-nf', '--number_feats', default=256, type=int)
    parser.add_argument('-csm', '--coord_sigmoid', action='store_true')
    parser.add_argument('-whsm', '--wh_sigmoid', action='store_true')
    parser.add_argument('-ims', '--image_size', default=512, type=int)
    parser.add_argument('-bpc', '--boxes_per_cell', default=2, type=int)
    parser.add_argument('-l2', '--l2_regularization', action='store_true', help='use l2 regularization')
    parser.add_argument('-l2f', '--l2_factor', default=5e-3, type=float)
    parser.add_argument('-bhm', '--bbox_hm', action='store_true', help='use focal loss')
    parser.add_argument('-bhml', '--bbox_hm_level', default=1, type=int, choices=[i for i in range(9)])
    # parser.add_argument('-cs', '--csize', default=64, type=int)
    parser.add_argument('-fc', '--focal_loss', action='store_true', help='use focal loss')
    parser.add_argument('-lw', '--load_weights', action='store_true', help='load weighs from wights dir')
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--position', default="tail_down16", type=str,
                        choices=["tail", "tail_tsp", "tail_down4", "tail_tsp_self", "tail_down16_v2",
                                 "tail_conv_deep", "tail_conv_deep_fc", "tail_down8", "tail_down16"])
    parser.add_argument('--train_mode', default="all", type=str, choices=["all", "scope"])
    parser.add_argument('--restore_mode', default="all", type=str, choices=["all", "scope"])
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    parser.add_argument('--factor', default=0.3, type=float)
    parser.add_argument('--ob_f', default=6.0, type=float)
    parser.add_argument('--noob_f', default=0.6, type=float)
    parser.add_argument('--coo_f', default=2.0, type=float)
    parser.add_argument('--cl_f', default=40.0, type=float)
    parser.add_argument('-lr', '--learning_rate', default=2.5e-4, type=float)
    parser.add_argument('-lrd', '--learning_rate_decay', default=1, type=float)
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
