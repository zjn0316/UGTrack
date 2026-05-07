import torch
from torch.utils.data.distributed import DistributedSampler

from lib.train.dataset import OTB100UWB, UAV123UWB, CustomDataset
from lib.train.data import uwb_processing, uwb_sampler, LTRLoader, opencv_loader
from lib.train.data.loader import resolve_num_workers
from lib.train.data import transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    # ==================================
    # 基础训练参数同步
    # ==================================
    # 将配置中的训练与数据处理公共字段写入 settings。
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {"template": cfg.DATA.TEMPLATE.FACTOR,
                                   "search": cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {"template": cfg.DATA.TEMPLATE.SIZE,
                          "search": cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {"template": cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     "search": cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {"template": cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    "search": cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.uwb_seq_len = cfg.DATA.UWB.SEQ_LEN  # [UGTrack独立参数] UWB 时序长度


def names2datasets(name_list: list, settings, image_loader, split='train'):
    # ==================================
    # 数据集名称到实例的映射
    # ==================================
    # 根据配置中的数据集名称列表，构造对应的数据集对象。
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        # 限定当前支持的数据集名称范围，便于后续增量扩展。
        if name == 'OTB100_UWB':
            datasets.append(
                OTB100UWB(
                    settings.env.otb100_uwb_dir,
                    split=split,
                    image_loader=image_loader,
                    uwb_seq_len=settings.uwb_seq_len,
                )
            )
        elif name == 'UAV123_UWB':
            datasets.append(
                UAV123UWB(
                    settings.env.uav123_uwb_dir,
                    split=split,
                    image_loader=image_loader,
                    uwb_seq_len=settings.uwb_seq_len,
                )
            )
        elif name == 'CUSTOM_DATASET':
            datasets.append(
                CustomDataset(
                    settings.env.custom_dataset_dir,
                    split=split,
                    image_loader=image_loader,
                    uwb_seq_len=settings.uwb_seq_len,
                )
            )
        else:
            raise ValueError(f'Unknown dataset: {name}')
    return datasets


def build_dataloaders(cfg, settings):
    stage = int(getattr(cfg.TRAIN, 'STAGE', 1))
    num_workers = resolve_num_workers(cfg.TRAIN.NUM_WORKER)
    # ==================================
    # 数据增强与变换配置
    # ==================================
    # 与 OSTrack 保持一致：joint 变换使用灰度增强 + 水平翻转。
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    # ==================================
    # 预处理模块构建
    # ==================================
    # 基于模板与搜索区域配置构建训练/验证预处理流程。
    data_processing_train = uwb_processing.UWBProcessing(search_area_factor=search_area_factor,
                                                         output_sz=output_sz,
                                                         center_jitter_factor=settings.center_jitter_factor,
                                                         scale_jitter_factor=settings.scale_jitter_factor,
                                                         mode='sequence',
                                                         transform=transform_train,
                                                         joint_transform=transform_joint,
                                                         settings=settings)

    data_processing_val = uwb_processing.UWBProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_val,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    # ==================================
    # 训练集采样器与加载器
    # ==================================
    # 从配置读取模板帧数量、搜索帧数量与采样模式，构建训练数据管线。
    settings.num_template = getattr(cfg.DATA.TEMPLATE, 'NUMBER', 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, 'NUMBER', 1)
    sampler_mode = getattr(cfg.DATA, 'SAMPLER_MODE', 'causal')
    train_cls = getattr(cfg.TRAIN, 'TRAIN_CLS', False)
    print('sampler_mode', sampler_mode)

    dataset_train = uwb_sampler.UWBTrackingSampler(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader, split='train'),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing_train,
        frame_sample_mode=sampler_mode,
        train_cls=train_cls,
    )

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=num_workers, drop_last=True, stack_dim=1, sampler=train_sampler)

    # ==================================
    # 验证集采样器与加载器
    # ==================================
    # 验证集与训练集保持相同的采样逻辑与处理流程。
    dataset_val = uwb_sampler.UWBTrackingSampler(
        datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader, split='val'),
        p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing_val,
        frame_sample_mode=sampler_mode,
        train_cls=train_cls,
    )

    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=num_workers, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    # ==================================
    # 参数设置
    # ==================================
    # stage=2：联合跟踪训练，按 tracker.backbone 与其余参数分组。
    # tracker.backbone 使用更小学习率进行微调，其余参数使用基础学习率。
    if int(getattr(cfg.TRAIN, 'STAGE', 1)) == 2:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters()
                        if 'tracker.backbone' not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters()
                           if 'tracker.backbone' in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if is_main_process():
            print('Learnable parameters are shown below.')
    # stage=1：UWB 预训练阶段，直接优化所有可训练参数。
    else:
        param_dicts = [{"params": [p for p in net.parameters() if p.requires_grad]}]

    # ====================================
    # 创建优化器。
    # ====================================
    # 当前仅支持 AdamW 优化器。
    if cfg.TRAIN.OPTIMIZER == 'ADAMW':
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )
    else:
        raise ValueError('Unsupported Optimizer')

    # ====================================
    # 创建学习率调度器。
    # ====================================
    # StepLR 每隔 LR_DROP_EPOCH 个 epoch 降低一次学习率。
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    else:
        raise ValueError('Unsupported scheduler')

    # 返回优化器与学习率调度器，供训练器统一管理。
    return optimizer, lr_scheduler
