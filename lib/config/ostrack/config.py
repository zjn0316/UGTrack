from easydict import EasyDict as edict
import yaml

"""
Add default config for OSTrack.
"""
cfg = edict()

# DATA
cfg.DATA = edict()                                                        # 数据配置对象
cfg.DATA.MEAN = [0.485, 0.456, 0.406]                                     # 图像归一化均值
cfg.DATA.STD = [0.229, 0.224, 0.225]                                      # 图像归一化标准差
cfg.DATA.SAMPLER_MODE = "causal"                                          # 采样模式：因果采样
cfg.DATA.MAX_SAMPLE_INTERVAL = 200                                        # 最大帧采样间隔
# DATA.TRAIN
cfg.DATA.TRAIN = edict()                                                  # 训练数据配置对象
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]               # 训练数据集名称列表
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]                                    # 训练数据集采样比例
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000                                   # 每个epoch的样本数量
# DATA.VAL
cfg.DATA.VAL = edict()                                                    # 验证数据配置对象
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]                            # 验证数据集名称列表
cfg.DATA.VAL.DATASETS_RATIO = [1]                                         # 验证数据集采样比例
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000                                     # 每个epoch的验证样本数量
# DATA.SEARCH
cfg.DATA.SEARCH = edict()                                                 # 搜索区域配置对象
cfg.DATA.SEARCH.NUMBER = 1                                                # 搜索区域数量
cfg.DATA.SEARCH.SIZE = 256                                                # 搜索区域图像尺寸
cfg.DATA.SEARCH.FACTOR = 4.0                                              # 搜索区域扩展因子
cfg.DATA.SEARCH.CENTER_JITTER = 3                                       # 搜索区域中心抖动范围
cfg.DATA.SEARCH.SCALE_JITTER = 0.5                                        # 搜索区域尺度抖动范围
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()                                               # 模板配置对象
cfg.DATA.TEMPLATE.NUMBER = 1                                              # 模板数量
cfg.DATA.TEMPLATE.SIZE = 128                                              # 模板图像尺寸
cfg.DATA.TEMPLATE.FACTOR = 2.0                                            # 模板扩展因子
cfg.DATA.TEMPLATE.CENTER_JITTER = 0                                       # 模板中心抖动范围
cfg.DATA.TEMPLATE.SCALE_JITTER = 0                                        # 模板尺度抖动范围

# MODEL
cfg.MODEL = edict()                                                      # 模型配置对象
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"                    # 预训练模型文件路径
cfg.MODEL.EXTRA_MERGER = False                                           # 是否使用额外融合层
cfg.MODEL.RETURN_INTER = False                                           # 是否返回中间层特征
cfg.MODEL.RETURN_STAGES = []                                             # 指定返回的特征阶段列表
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()                                             # 主干网络配置对象
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"                         # 主干网络类型
cfg.MODEL.BACKBONE.STRIDE = 16                                           # 主干网络步长
cfg.MODEL.BACKBONE.MID_PE = False                                        # 是否使用中间位置编码
cfg.MODEL.BACKBONE.SEP_SEG = False                                       # 是否分离分割头
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'                                   # 特征拼接模式
cfg.MODEL.BACKBONE.MERGE_LAYER = 0                                       # 特征融合层索引
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False                                 # 是否添加CLS token
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'                         # CLS token使用模式
cfg.MODEL.BACKBONE.CE_LOC = []                                           # Candidate Elimination模块位置列表
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []                                    # Candidate Elimination保留比例列表
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'                             # CE模板范围：ALL/CTR_POINT/CTR_REC/GT_BOX
# MODEL.HEAD
cfg.MODEL.HEAD = edict()                                                 # 检测头配置对象
cfg.MODEL.HEAD.TYPE = "CENTER"                                           # 检测头类型
cfg.MODEL.HEAD.NUM_CHANNELS = 256                                        # 检测头通道数

# TRAIN
cfg.TRAIN = edict()                                                      # 训练配置对象
cfg.TRAIN.LR = 0.0004                                                    # 学习率
cfg.TRAIN.WEIGHT_DECAY = 0.0001                                          # 权重衰减系数
cfg.TRAIN.EPOCH = 300                                                    # 训练总轮数
cfg.TRAIN.LR_DROP_EPOCH = 240                                            # 学习率下降的epoch
cfg.TRAIN.BATCH_SIZE = 16                                                # 批次大小
cfg.TRAIN.NUM_WORKER = 8                                                 # 数据加载工作进程数
cfg.TRAIN.OPTIMIZER = "ADAMW"                                            # 优化器类型
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1                                      # 主干网络学习率乘数
cfg.TRAIN.GIOU_WEIGHT = 2.0                                              # GIoU损失权重
cfg.TRAIN.L1_WEIGHT = 5.0                                                # L1损失权重
cfg.TRAIN.FREEZE_LAYERS = [0, ]                                          # 需要冻结的层索引列表
cfg.TRAIN.PRINT_INTERVAL = 50                                            # 日志打印间隔
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20                                        # 验证间隔epoch
cfg.TRAIN.GRAD_CLIP_NORM = 0.1                                           # 梯度裁剪范数
cfg.TRAIN.AMP = False                                                    # 是否启用自动混合精度训练

cfg.TRAIN.CE_START_EPOCH = 20                                            # Candidate Elimination起始epoch
cfg.TRAIN.CE_WARM_EPOCH = 80                                             # Candidate Elimination预热epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1                                           # ViT主干网络的Drop Path比率

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()                                            # 学习率调度器配置对象
cfg.TRAIN.SCHEDULER.TYPE = "step"                                        # 调度器类型
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1                                     # 学习率衰减率

# TEST
cfg.TEST = edict()                                                       # 测试配置对象
cfg.TEST.TEMPLATE_FACTOR = 2.0                                           # 测试时模板扩展因子
cfg.TEST.TEMPLATE_SIZE = 128                                             # 测试时模板尺寸
cfg.TEST.SEARCH_FACTOR = 4.0                                             # 测试时搜索区域扩展因子
cfg.TEST.SEARCH_SIZE = 256                                               # 测试时搜索区域尺寸
cfg.TEST.EPOCH = 300                                                     # 测试使用的checkpoint epoch


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename, 'r', encoding='utf-8') as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
