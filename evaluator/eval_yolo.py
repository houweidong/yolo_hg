from Eutils.pathmagic import context
with context():
    import argparse
    import numpy as np
    from model.hourglass_yolo_net_multi_gpu import HOURGLASSYOLONet
    from evaluator.Eutils.pascal_val import PASCAL_VAL
    # from evaluator.Eutils.coco_val import COCO_VAL
    from evaluator.Eutils.detector import Detector
    import utils.config as cfg
    from utils.logger import Logger
    from utils.config_utils import get_config,ds_config
    from tqdm import tqdm
    import tensorflow as tf
    import copy
    import os


# import cv2
# from evaluator.Eutils.draw_result import draw_result


class EVALUATOR(object):

    def __init__(self, detector, data):
        self.detector = detector
        self.data = data
        self.gt = self.data.gt
        self.image_ids, self.bboxes, \
        self.prob, self.annotations = self.prepare()
        self.precision, self.recall = self.pr_curve()

    def prepare(self):
        image_ids, bboxes, prob = [], [], []
        annotations = {}
        # while img_batch:
        for i in tqdm(range(self.data.num_batch), desc='batch forward'):
            # print("{:5}th batch".format(i))
            img_batch, bbox_batch = self.data.get_batch()
            results = self.detector.detect_batch(img_batch)
            for ii in range(len(results)):
                boxes_filtered, probs_filtered = results[ii]
                # bbox_gt = bbox_batch[ii]['bbox_det']['bboxes']
                # filter_mat_probs = np.array(probs_filtered >= cfg.THRESHOLD, dtype='bool')
                # filter_mat_probs = np.nonzero(filter_mat_probs)
                # boxes_ft_prob = boxes_filtered[filter_mat_probs]
                # probs_ft_prob = probs_filtered[filter_mat_probs]
                # image = img_batch[ii]
                # draw_result(image, bbox_gt, (0, 0, 255))
                # draw_result(image, boxes_ft_prob, (255, 0, 0))
                # cv2.imshow('Image', image)
                # cv2.waitKey(0)
                image_ids.extend([bbox_batch[ii]['id']] * len(boxes_filtered))
                bboxes.extend(boxes_filtered)
                prob.extend(probs_filtered)
                if bbox_batch[ii]['id'] not in annotations:
                    annotations[bbox_batch[ii]['id']] = copy.deepcopy(bbox_batch[ii]['bbox_det'])
        sorted_ind = np.argsort(prob)[::-1]
        sorted_prob = np.sort(prob)[::-1]
        BB = np.array(bboxes)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        return image_ids, BB, sorted_prob, annotations

    def pr_curve(self):
        nd = len(self.image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in tqdm(range(nd), desc='painting PR curve'):
            # for d in range(nd):
            R = self.annotations[self.image_ids[d]]
            bb = self.bboxes[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bboxes'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0] - BBGT[:, 2] / 2, bb[0] - bb[2] / 2)
                iymin = np.maximum(BBGT[:, 1] - BBGT[:, 3] / 2, bb[1] - bb[3] / 2)
                ixmax = np.minimum(BBGT[:, 0] + BBGT[:, 2] / 2, bb[0] + bb[2] / 2)
                iymax = np.minimum(BBGT[:, 1] + BBGT[:, 3] / 2, bb[1] + bb[3] / 2)
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = bb[2] * bb[3] + BBGT[:, 2] * BBGT[:, 3] - inters

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > cfg.IOU_THRESHOLD_GT:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(self.gt)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth

        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        return prec, rec

    def eval(self, use_07_metric=False):
        """ ap = eval(rec, prec, [use_07_metric])
        Compute AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """

        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(self.recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(self.precision[self.recall >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], self.recall, [1.]))
            mpre = np.concatenate(([0.], self.precision, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ims', '--image_size', default=512, type=int)
    parser.add_argument('-g','--gpu', type=str)
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    parser.add_argument('-ds', '--data_source', default='all', type=str, choices=['coco', 'pascal', 'all'])
    parser.add_argument('-ef', '--eval_file', type=str, required=True)
    parser.add_argument('-lf', '--log_file', type=str)
    parser.add_argument('-al', '--auto_all', action='store_true')
    # when calculate single model
    parser.add_argument('--weights', default="hg_yolo-240000", type=str)
    parser.add_argument('--weight_dir', default='../log_bbox_hm/0.8_0.08_0.03_conv_fc_l2_0.005_bhm5', type=str)
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if not args.auto_all:
        strings = get_config(args.weight_dir)

        net = HOURGLASSYOLONet('eval')
        detector = Detector(net, os.path.join(args.weight_dir, args.weights))
        # data = COCO_VAL()
        data = PASCAL_VAL()
        evaluator = EVALUATOR(detector, data)
        ap = evaluator.eval()
        log = Logger(args.eval_file, level='debug')
        log.logger.info('\n calculate single ap from {} {}\n'.format(args.weight_dir, args.weights))
        log.logger.info('Data sc:{}  AP:{}  Weights:{}  {}'.format(
            data.__class__.__name__, ap, args.weights, strings))
    else:
        data_source = ds_config(args)
        log = Logger(args.eval_file, level='debug')
        log.logger.info('\n calculate ap from {}\n'.format(args.eval_file))
        model_start = 'hg_yolo'
        rootdir = '../' + args.log_file
        root_list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        root_list.sort()
        for path in root_list:
            model_dir = os.path.join(rootdir, path)
            models = os.listdir(model_dir)
            models = filter(lambda x: x.startswith(model_start), models)
            models = list(set(map(lambda x: x.split('.')[0], models)))
            models.sort(key=lambda x: int(x[8:]))
            for data in data_source:
                for model in models:
                    strings = get_config(model_dir)
                    tf.reset_default_graph()
                    net = HOURGLASSYOLONet('eval')
                    detector = Detector(net, os.path.join(model_dir, model))
                    evaluator = EVALUATOR(detector, data)
                    ap = evaluator.eval()
                    log.logger.info('Data sc:{}  AP:{:<5.5f}  Weights:{}  {}'.format(
                        data.__class__.__name__, ap, model, strings))
                    detector.sess.close()
                    del net
                    del detector
                    del evaluator


if __name__ == '__main__':
    main()
    # print(os.path.realpath('.'))
    # print(os.path.dirname(os.path.realpath('.')))
    # print(os.sep)
    #
    # print(os.path.dirname(os.path.realpath('.')).split(os.sep))

