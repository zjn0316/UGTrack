from easydict import EasyDict as edict
import yaml


cfg = edict()

# Data / 数据
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.SAMPLER_MODE = "causal"
cfg.DATA.MAX_SAMPLE_INTERVAL = 200

cfg.DATA.UWB = edict()
cfg.DATA.UWB.SEQ_LEN = 10

cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["OTB100_UWB"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 5000

cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["OTB100_UWB"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 1000

cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3
cfg.DATA.SEARCH.SCALE_JITTER = 0.5

cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# Model / 模型
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []
cfg.MODEL.NUM_OBJECT_QUERIES = 1

cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = "direct"
cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = "ALL"

cfg.MODEL.BACKBONE.UWB_ENCODER = "tcn"
cfg.MODEL.BACKBONE.UWB_INPUT_DIM = 2
cfg.MODEL.BACKBONE.UWB_EMBED_DIM = 128
cfg.MODEL.BACKBONE.UWB_MLP_HIDDEN_DIMS = [128, 128]
cfg.MODEL.BACKBONE.UWB_MLP_DROPOUT = 0.1
cfg.MODEL.BACKBONE.UWB_GRU_INPUT_PROJ_DIM = 64
cfg.MODEL.BACKBONE.UWB_GRU_HIDDEN_DIM = 128
cfg.MODEL.BACKBONE.UWB_GRU_DROPOUT = 0.1
cfg.MODEL.BACKBONE.UWB_TCN_CHANNELS = 64
cfg.MODEL.BACKBONE.UWB_TCN_DILATIONS = [1, 2, 4]
cfg.MODEL.BACKBONE.UWB_TCN_KERNEL_SIZE = 3
cfg.MODEL.BACKBONE.UWB_TCN_DROPOUT = 0.1
cfg.MODEL.BACKBONE.UWB_PRUNE_ENABLE = False
cfg.MODEL.BACKBONE.UWB_PRUNE_KEEP_RATIO = 0.5
cfg.MODEL.BACKBONE.UWB_PRUNE_MIN_KEEP_RATIO = 0.25
cfg.MODEL.BACKBONE.UWB_PRUNE_MAX_KEEP_RATIO = 1.0
cfg.MODEL.BACKBONE.UWB_PRUNE_CONF_DYNAMIC = True

cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256
cfg.MODEL.HEAD.UWB_TOKEN_HEAD = "mlp"
cfg.MODEL.HEAD.UWB_TOKEN_DIM = 768
cfg.MODEL.HEAD.UWB_HEAD_DROPOUT = 0.1
cfg.MODEL.HEAD.UWB_PRED_MODE = "residual"

# Train / 训练
cfg.TRAIN = edict()
cfg.TRAIN.STAGE = 1
cfg.TRAIN.LR = 0.001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 100
cfg.TRAIN.LR_DROP_EPOCH = 70
cfg.TRAIN.BATCH_SIZE = 256
cfg.TRAIN.NUM_WORKER = 0
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.UWB_COORD_LOSS = "l1"
cfg.TRAIN.UWB_CONF_LOSS = "bce"
cfg.TRAIN.UWB_PRED_WEIGHT = 1.0
cfg.TRAIN.UWB_CONF_WEIGHT = 0.5
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 10
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.CE_START_EPOCH = 20
cfg.TRAIN.CE_WARM_EPOCH = 80
cfg.TRAIN.DROP_PATH_RATE = 0.1

cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"

# Test / 测试
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.EPOCH = 100


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for key, value in src_edict.items():
            if isinstance(value, edict):
                dest_dict[key] = {}
                _edict2dict(dest_dict[key], value)
            else:
                dest_dict[key] = value


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, dict):
        for key, value in exp_cfg.items():
            if key not in base_cfg:
                raise ValueError("{} not exist in config.py".format(key))

            if isinstance(value, dict):
                _update_config(base_cfg[key], value)
            else:
                base_cfg[key] = value


def update_config_from_file(filename, base_cfg=None):
    with open(filename, "r", encoding="utf-8") as f:
        exp_config = edict(yaml.safe_load(f))

    if base_cfg is not None:
        _update_config(base_cfg, exp_config)
    else:
        _update_config(cfg, exp_config)
