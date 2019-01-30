import os

# ------------------------------------------------------------
# path and dataset parameter
OUTPUT_DIR = 'log_wh_sigmoid_bl_down'
OUTPUT_DIR_TASK = None
WEIGHTS_DIR = 'weights'
WEIGHTS_FILE = None
CHECKPOINT_EXCLUDE_SCOPES = []
TRAINABLE_SCOPES = []
PASCAL_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                  'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                  'train', 'tvmonitor']
# COCO_CLASSES = ['noperson', 'person']
COCO_CLASSES = ['person']

# coco parameter
COCO_MAX_PERSON_PER_PIC = 13
COCO_MAX_OBJECT_PER_PIC = 100
COCO_EXAMPLES = 117266
COCO_BATCH_SIZE = 20
COCO_EPOCH_SIZE = COCO_EXAMPLES // COCO_BATCH_SIZE
# COCO_TRAIN_FILENAME = '/root/dataset/tfrecord_big_person/train/'
# COCO_VAL_FILENAME = '/root/dataset/tfrecord_big_person/val/'
COCO_TRAIN_FILENAME = '/home/new/dataset/tfrecord1/train/'
COCO_VAL_FILENAME = '/home/new/dataset/tfrecord1/val/'
# ------------------------------------------------------------


# ------------------------------------------------------------
# model parameter
L2 = True
L2_FACTOR = 0.5
LOSS_FACTOR = 0.1
# support middle | tail | tail_cov | tail_tsp | tail_tsp_self | tail_conv_deep | tail_conv_deep_fc
ADD_YOLO_POSITION = "tail_down16_v2"
# all: train all var in net  sp: train var in TRAINABLE_SCOPES list
TRAIN_MODE = 'all'
RESTORE_MODE = 'all'
IMAGE_SIZE = 256
CELL_SIZE = 64
BOXES_PER_CELL = 2
# DISP_CONSOLE = False

COORD_SIGMOID = False
WH_SIGMOID = False
BOX_FOCAL_LOSS = False
# For yolo
BOX_HOT_MAP = False
BOX_HOT_MAP_LEVEL = 0
# For keypoints
HG_HOT_MAP_DIFF_LEVEL = 0
HG_HOT_MAP_LEVEL = 1

OBJECT_SCALE = 20.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 40.0
COORD_SCALE = 100.0

# focal loss
R_OBJECT = 2
# ALPHA_OBJECT = 5

# ....hourglass parameter
# WIDTH = IMAGE_SIZE
# HEIGHT = IMAGE_SIZE
# HM_SIZE = IMAGE_SIZE // 4
# HM_WIDTH = IMAGE_SIZE // 4
COCO_NPOINTS = 17
HG_CELL_SIZE = IMAGE_SIZE // 4
NUM_MOUDEL = 1  # hourglass 中residual 模块的数量
NUM_STACK = 2  # hourglass 堆叠的层数
NUM_FEATS = 256  # hourglass 中特征图的数量
# ------------------------------------------------------------


# ------------------------------------------------------------
# solver parameter
GPU = '0'
GPU_NUMBER = 1
LEARNING_RATE = 2.5e-4
DECAY_STEPS = 10000
DECAY_RATE = 1
STAIRCASE = True
MAX_ITER = 300000
SUMMARY_ITER = 10
SAVE_ITER = 10000
# ------------------------------------------------------------


# ------------------------------------------------------------
# test parameter

THRESHOLD = 0.8
IOU_THRESHOLD_NMS = 0.2
HG_THRESHOLD = 0.2
HG_FACTOR_HEIGHT = 1.3
HG_FACTOR_WIDTH = 0.8

IOU_THRESHOLD_GT = 0.5
# COCO_ANNOTATION_FILE = '/root/dataset/annotations_trainval2017/annotations/instances_val2017.json'
# COCO_VAL_IMAGE_FILE = '/root/dataset/val2017/'
COCO_ANNOTATION_FILE = '/home/new/dataset/annotations_trainval2017/annotations/instances_val2017.json'
COCO_VAL_IMAGE_FILE = '/home/new/dataset/val2017/'

# PASCAL_PATH = '/root/dataset/data/pascal_voc/'
PASCAL_PATH = '/home/new/dataset/data/pascal_voc/'
PASCAL_DATA = os.path.join(PASCAL_PATH, 'VOCdevkit/VOC2012/')
# ------------------------------------------------------------
