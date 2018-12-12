import os

# ------------------------------------------------------------
# path and dataset parameter
#DATA_PATH = 'data'
#PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')
#CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')
OUTPUT_DIR = 'output'
WEIGHTS_DIR = 'weights'
WEIGHTS_FILE = None
CHECKPOINT_EXCLUDE_SCOPES = ['conv_pad3', 'batch_norm_relu', 'residual1', 'down_sampling', 'residual2',
                             'residual3', 'hourglass', 'residual', 'lin', 'conv_same', 'next_input',
                             'yolo/conv_64_2_52_1', 'yolo/conv_64_2_52_2', 'yolo/fc_36']
TRAINABLE_SCOPES = ['conv_pad3', 'batch_norm_relu', 'residual1', 'down_sampling', 'residual2', 'residual3',
                    'hourglass', 'residual', 'lin', 'conv_same', 'next_input',
                    'yolo/conv_64_2_52_1', 'yolo/conv_64_2_52_2', 'yolo/fc_36']
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
COCO_CLASSES = ['person']
FLIPPED = True
# ....hourglass parameter
#FILENAME = '/root/dataset/final_train_512.tfrecords'
COCO_FILENAME = '/root/dataset/coco_train_181209.tfrecords'
# ------------------------------------------------------------


# ------------------------------------------------------------
# model parameter
ADD_YOLO_POSITION = "tail"
IMAGE_SIZE = 256
CELL_SIZE = 64
BOXES_PER_CELL = 2
KEEP_PROB = 0.5
ALPHA = 0.1
DISP_CONSOLE = False

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0

# ....hourglass parameter
WIDTH = 512
HEIGHT = 512
HM_HEIGHT = 64
HM_WIDTH = 64
WHAT_FACK = 10
NPOINTS = 16
COCO_NPOINTS = 17
HG_CELL_SIZE = 64
# ------------------------------------------------------------


# ------------------------------------------------------------
# solver parameter
BATCH_SIZE = 20
GPU = '0'
LEARNING_RATE = 1e-4
DECAY_STEPS = 5000
DECAY_RATE = 0.1
STAIRCASE = True
MAX_ITER = 100000
SUMMARY_ITER = 10
SAVE_ITER = 5000
# coco parameter
COCO_EXAMPLES = 7843
COCO_BATCH_SIZE = 15
COCO_EPOCH_SIZE = COCO_EXAMPLES // COCO_BATCH_SIZE
COCO_LEARNING_RATE = 1e-4
COCO_DECAY_STEPS = 5000
COCO_DECAY_RATE = 0.1
COCO_STAIRCASE = True
COCO_MAX_ITER = 100000
COCO_SUMMARY_ITER = 10
COCO_SAVE_ITER = 5000

# ....hourglass parameter
HOURGLASS_BATCH_SIZE = 4
EPOCH_SIZE = 2500
LEARNING_RATE_HG = 2.5e-4
DECAY_STEPS_HG = 10000
DECAY_RATE_HG = 0.1
STAIRCASE_HG = True
# SAVE_ITER_HG = 500
# SUMMARY_ITER_HG = 300
# ------------------------------------------------------------


# ------------------------------------------------------------
# test parameter
THRESHOLD = 0.02
IOU_THRESHOLD = 0.5
# ------------------------------------------------------------
