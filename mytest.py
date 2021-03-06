import os
import collections
import utils.config as cfg


def get_config(config_path):
    config = os.path.join(config_path, 'config.txt')
    values = collections.OrderedDict()
    keys = ['ADD_YOLO_POSITION', 'IMAGE_SIZE', 'CELL_SIZE', 'BOXES_PER_CELL', 'LOSS_FACTOR', 'LEARNING_RATE',
            'OBJECT_SCALE', 'NOOBJECT_SCALE', 'COORD_SCALE',
            'BOX_FOCAL_LOSS', 'BOX_HOT_MAP_LEVEL', 'L2', 'L2_FACTOR', 'COORD_SIGMOID', 'WH_SIGMOID']
    values = values.fromkeys(keys)
    for line in open(config):
        name, value = line.split(': ')[0], line.split(': ')[1]
        if name in keys:
            values[name] = value.strip()
    cfg.ADD_YOLO_POSITION = values['ADD_YOLO_POSITION']
    cfg.IMAGE_SIZE = int(values['IMAGE_SIZE'])
    cfg.CELL_SIZE = int(values['CELL_SIZE'])
    cfg.L2 = False
    cfg.L2 = bool(values['L2'])
    cfg.L2_FACTOR = float(values['L2_FACTOR'])
    cfg.BOX_FOCAL_LOSS = bool(values['BOX_FOCAL_LOSS'])
    cfg.BOX_HOT_MAP_LEVEL = int(values['BOX_HOT_MAP_LEVEL'])
    cfg.BOXES_PER_CELL = int(values['BOXES_PER_CELL'])
    cfg.COORD_SIGMOID = bool(values['COORD_SIGMOID'])
    cfg.WH_SIGMOID = bool(values['WH_SIGMOID'])
    strings = config_path.split('/')[2] + '  '
    for i, value in values.items():
        strings += '{}:{}  '.format(i, value)
    return strings


def update_config(args):
    if args.gpu is not None:
        cfg.GPU = args.gpu
    if args.cpu:
        cfg.GPU = ''
    if args.log_dir:
        cfg.OUTPUT_DIR_TASK = args.log_dir

    size_dive4 = ["tail", "tail_tsp", "tail_conv", "tail_tsp_self",
                  "tail_conv_deep", "tail_conv_deep_fc"]
    size_dive8 = ["tail_conv_32"]
    size_dive16 = ["tail_conv_16"]
    cfg.IMAGE_SIZE = args.image_size
    cfg.ADD_YOLO_POSITION = args.position
    if args.position in size_dive4:
        cfg.CELL_SIZE = args.image_size // 4
    elif args.position in size_dive8:
        cfg.CELL_SIZE = args.image_size // 8
    elif args.position in size_dive16:
        cfg.CELL_SIZE = args.image_size // 16
    cfg.TRAIN_MODE = args.train_mode
    cfg.RESTORE_MODE = args.restore_mode
    if args.load_weights:
        # update_config_paths(args.data_dir, args.weights)
        cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, args.weights)
    cfg.L2 = args.l2_regularization
    cfg.L2_FACTOR = args.l2_factor
    cfg.LOSS_FACTOR = args.factor
    cfg.OBJECT_SCALE = args.ob_f
    cfg.NOOBJECT_SCALE = args.noob_f
    cfg.COORD_SCALE = args.coo_f
    cfg.CLASS_SCALE = args.cl_f
    cfg.BOX_HOT_MAP = args.bbox_hm
    cfg.BOX_HOT_MAP_LEVEL = args.bbox_hm_level
    # cfg.BOX_HOT_MAP_LEVEL = args.bhmlevel
    cfg.BOX_FOCAL_LOSS = args.focal_loss
    cfg.BOXES_PER_CELL = args.boxes_per_cell
    cfg.COORD_SIGMOID = args.coord_sigmoid
    cfg.LEARNING_RATE = args.learning_rate
    cfg.DECAY_RATE = args.learning_rate_decay
    cfg.WH_SIGMOID = args.wh_sigmoid
    cfg.GPU_NUMBER = len(list(filter(None, args.gpu.split(','))))

    print("YOLO POSITION: {}".format(cfg.ADD_YOLO_POSITION))
    print("LOSS_FACTOR:{}  OB_SC: {}  NOOB_SC: {}  "
          "COO_SC: {}  CL_SC: {}  BHP: {}  BHPL: {}  "
          "L2: {}  L2_F: {}  IMAGE_SIZE: {}  COORDSM: {}  WHSM: {}".
          format(args.factor, args.ob_f, args.noob_f, args.coo_f, args.cl_f,
                 args.bbox_hm, args.bbox_hm_level, args.l2_regularization, args.l2_factor,
                 args.image_size, args.coord_sigmoid, args.wh_sigmoid))
    print("LR: {}  LRD: {}".format(args.learning_rate, args.learning_rate_decay))
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
