import os
import collections
import utils.config as cfg


def get_config(config_path):
    config = os.path.join(config_path, 'config.txt')
    values = collections.OrderedDict()
    keys = ['ADD_YOLO_POSITION', 'CELL_SIZE', 'LOSS_FACTOR', 'LEARNING_RATE',
            'OBJECT_SCALE', 'NOOBJECT_SCALE', 'COORD_SCALE',
            'BOX_FOCAL_LOSS', 'BOX_HOT_MAP_LEVEL', 'L2', 'L2_FACTOR']
    values = values.fromkeys(keys)
    for line in open(config):
        name, value = line.split(': ')[0], line.split(': ')[1]
        if name in keys:
            values[name] = value.strip()
    cfg.ADD_YOLO_POSITION = values['ADD_YOLO_POSITION']
    cfg.CELL_SIZE = int(values['CELL_SIZE'])
    cfg.L2 = False
    # if values['L2']:
    cfg.L2 = bool(values['L2'])
    # if values['L2']:
    cfg.L2_FACTOR = float(values['L2_FACTOR'])
    cfg.BOX_FOCAL_LOSS = bool(values['BOX_FOCAL_LOSS'])
    # if values['BOX_HOT_MAP_LEVEL']:
    cfg.BOX_HOT_MAP_LEVEL = int(values['BOX_HOT_MAP_LEVEL'])
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

    cfg.ADD_YOLO_POSITION = args.position
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
    cfg.CELL_SIZE = args.csize
    cfg.BOX_HOT_MAP = args.bbox_hm
    cfg.BOX_HOT_MAP_LEVEL = args.bbox_hm_level
    # cfg.BOX_HOT_MAP_LEVEL = args.bhmlevel
    cfg.BOX_FOCAL_LOSS = args.focal_loss

    print("YOLO POSITION: {}".format(cfg.ADD_YOLO_POSITION))
    print("LOSS_FACTOR:{}  OB_SC: {}  NOOB_SC: {}  "
          "COO_SC: {}  CL_SC: {}  BHP: {}  BHPL: {}  L2: {}  L2_F: {}".
          format(args.factor, args.ob_f, args.noob_f, args.coo_f, args.cl_f,
                 args.bbox_hm, args.bbox_hm_level, args.l2_regularization, args.l2_factor))
    print("LR: {}".format(cfg.LEARNING_RATE))
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
