# 系统与标准库导入
import importlib
import os

# PyTorch/计算库导入
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.parallel import DistributedDataParallel as DDP

# 项目本地模块导入
# 模型构建
from lib.models.ugtrack import build_ugtrack
# 训练框架相关
from lib.train.actors import UGTrackActor
from lib.train.trainers import LTRTrainer
# 损失函数
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss
# 基础公共函数
from .base_functions_ugtrack import build_dataloaders, get_optimizer_scheduler, update_settings


# ====================
# 训练运行函数 (run)
# ====================
# 加载配置、数据集加载、创建网络、移动到GPU、UGTrackActor、LTRTrainer、train
def run(settings):
    settings.description = "UGTrack training"

    # ====================
    # 配置文件加载与设置
    # ====================
    if not os.path.exists(settings.cfg_file):
        raise ValueError("{} doesn't exist.".format(settings.cfg_file))

    # 动态导入对应的配置模块（lib.config.ugtrack.config）
    config_module = importlib.import_module("lib.config.{}".format(settings.script_name) + ".config")
    cfg = config_module.cfg
    # 使用指定的 yaml 文件内容更新默认配置
    config_module.update_config_from_file(settings.cfg_file)

    # ====================
    # 训练阶段控制（当前仅支持阶段1与阶段2）
    # ====================
    stage = int(cfg.TRAIN.STAGE)
    if stage not in [1, 2]:
        raise NotImplementedError("train_script_ugtrack currently supports TRAIN.STAGE in [1, 2]")

    # 根据 cfg 更新 settings
    update_settings(settings, cfg)

    # ====================
    # 记录训练日志
    # ====================
    log_dir = os.path.join(settings.save_dir, "logs")
    if settings.local_rank in [-1, 0] and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 设置日志文件路径，格式为 "logs/<script_name>-<config_name>.log"
    settings.log_file = os.path.join(log_dir, "{}-{}.log".format(settings.script_name, settings.config_name))

    
    # ====================
    # 1.构建数据加载器
    # ====================
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # ====================
    # 2.创建网络模型
    # ====================
    net = build_ugtrack(cfg, training=True)

    # ====================
    # 将网络转移至 GPU 内存
    # ====================
    net.cuda()
    if settings.local_rank != -1:
        # 分布式模式：将模型包装为 DistributedDataParallel (DDP)
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:{}".format(settings.local_rank))
    else:
        # 单卡模式：使用默认的 0 号 GPU
        settings.device = torch.device("cuda:0")

    # ====================
    # 按训练阶段设置损失函数与 Actors
    # ====================
    if stage == 1:
        coord_loss_name = str(getattr(cfg.TRAIN, "UWB_COORD_LOSS", "l1")).lower()
        conf_loss_name = str(getattr(cfg.TRAIN, "UWB_CONF_LOSS", "bce")).lower()

        if coord_loss_name == "l1":
            uwb_pred_loss = l1_loss
        elif coord_loss_name == "mse":
            uwb_pred_loss = mse_loss
        else:
            raise ValueError("Unsupported UWB_COORD_LOSS: {}".format(coord_loss_name))

        if conf_loss_name == "bce":
            uwb_conf_loss = BCEWithLogitsLoss()
        elif conf_loss_name == "mse":
            uwb_conf_loss = mse_loss
        else:
            raise ValueError("Unsupported UWB_CONF_LOSS: {}".format(conf_loss_name))

        # Loss dictionary / 损失函数字典
        objective = {'uwb_pred': uwb_pred_loss, 'uwb_conf': uwb_conf_loss}
        loss_weight = {'uwb_pred': cfg.TRAIN.UWB_PRED_WEIGHT, 'uwb_conf': cfg.TRAIN.UWB_CONF_WEIGHT}
    else:
        focal_loss = FocalLoss()
        # 损失函数字典
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}

    actor = UGTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    # ====================
    # 优化器与学习率调度器设置
    # ====================
    # 初始化优化器（Optimizer）和学习率调度器（LR Scheduler）
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    # ====================
    # 训练器设置
    # ====================
    # 是否启用AMP
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    # 实例化训练引擎 LTRTrainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # ====================
    # 启动训练循环
    # ====================
    # 阶段2支持加载第一阶段训练产物
    load_previous_ckpt = stage == 2 and hasattr(settings, "project_path_prv")
    if settings.local_rank in [-1, 0]:
        print("UGTrack weight loading plan:")
        print("  stage: {}".format(stage))
        print("  load_latest/current experiment resume: True")
        print("  load_previous/stage-1 init: {}".format(load_previous_ckpt))
        if load_previous_ckpt:
            print("  previous checkpoint project: {}".format(settings.project_path_prv))
        elif stage == 2:
            print("  previous checkpoint project: None")
        print("  model pretrain file: {}".format(cfg.MODEL.PRETRAIN_FILE))

    # ====================
    # 启动训练循环
    # ====================
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True, load_previous_ckpt=load_previous_ckpt)
    # load_latest: 是否尝试加载最新的 checkpoint 继续训练
    # fail_safe: 启用容错机制
    # load_previous_ckpt: 阶段2是否加载前序阶段 checkpoint
