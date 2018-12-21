import os
import argparse
import numpy as np
from utils import config as cfg
from model.hourglass_yolo_net import HOURGLASSYOLONet
from evaluator.coco_val import COCO_VAL
from evaluator.detector import Detector


class EVALUATOR(object):

    def __init__(self, detector, data):
        self.detector = detector
        self.data = data
        self.precision, self.recall = self.prepare_pr()

    def prepare_pr(self):
        precision = []
        recall = []

        return precision, recall

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
    parser.add_argument('--position',
                        default="tail",
                        type=str,
                        choices=["tail", "tail_tsp", "tail_conv", "tail_tsp_self"])
    parser.add_argument('--weights', default="hg_yolo-400000", type=str)
    parser.add_argument('--weight_dir', default='log/20_1_100_5e-4', type=str)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    cfg.ADD_YOLO_POSITION = args.position
    net = HOURGLASSYOLONet()
    detector = Detector(net, os.path.join(args.weight_dir, args.weights))

    data = COCO_VAL()
    evaluator = EVALUATOR(detector, data)
    evaluator.eval()


if __name__ == '__main__':
    main()
