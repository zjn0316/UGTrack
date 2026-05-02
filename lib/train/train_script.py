# 系统与标准库导入
import os
import importlib

# PyTorch/计算库导入
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import l1_loss
from torch.nn.parallel import DistributedDataParallel as DDP

# 项目本地模块导入
# 模型构建
from lib.models.ostrack import build_ostrack
# 训练框架相关
from lib.train.actors import OSTrackActor
from lib.train.trainers import LTRTrainer
# 损失函数
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss
# 基础公共函数
from .base_functions import *

# ====================
# 训练运行函数 (run)
# ====================
# 加载配置、数据集加载、创建网络、移动到GPU、OSTrackActor、LTRTrainer、train
def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # ====================
    # 配置文件加载与设置
    # ====================
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    # 动态导入对应的配置模块（lib.config.ostrack.config）
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    # 使用指定的 yaml 文件内容更新默认配置
    config_module.update_config_from_file(settings.cfg_file)

    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        # for key in cfg.keys():
        #     print("%s configuration:" % key, cfg[key])
        #     print('\n')
    # 根据 cfg 更新 settings
    update_settings(settings, cfg)

    # ====================
    # 记录训练日志
    # ====================
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    # 设置日志文件路径，格式为 "logs/<script_name>-<config_name>.log"
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # 构建数据加载器
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # ====================
    # 模型相关配置
    # ====================
    # 对于特定的主干网络类型，指定 checkpoint 保存目录
    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # 创建网络模型
    if settings.script_name == "ostrack":
        net = build_ostrack(cfg)
    else:
        raise ValueError("illegal script name")

    # ====================
    # 设备配置与模型包装
    # ====================
    # 将网络转移至 GPU 内存
    net.cuda()
    if settings.local_rank != -1:
        # 分布式模式：将模型包装为 DistributedDataParallel (DDP)
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # 如果需要，添加 syncBN 转换器
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        # 单卡模式：使用默认的 0 号 GPU
        settings.device = torch.device("cuda:0")

    # 从配置中提取训练相关标志位
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    
    # ====================
    # 损失函数与 Actors
    # ====================
    if settings.script_name == "ostrack":
        focal_loss = FocalLoss()
        # 损失函数字典
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = OSTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # ====================
    # 优化器与学习率调度器设置
    # ====================
    # 初始化优化器（Optimizer）和学习率调度器（LR Scheduler）
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    # ====================
    # 训练器设置
    # ====================
    # 检查是否启用自动混合精度训练 (Automatic Mixed Precision)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    # 实例化训练引擎 LTRTrainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # ====================
    # 启动训练循环
    # ====================
    # 开始训练指定的 Epoch 数量
    # load_latest: 是否尝试加载最新的 checkpoint 继续训练
    # fail_safe: 启用容错机制
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
