import importlib
import os
import argparse

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import l1_loss, mse_loss

import _init_paths
from lib.train.admin import settings as ws_settings
from lib.train.base_functions_ugtrack import build_dataloaders, get_optimizer_scheduler, update_settings
from lib.models.ugtrack import build_ugtrack
from lib.train.actors import UGTrackActor
from lib.train.trainers import LTRTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='UGTrack Stage-1 training')
    parser.add_argument('--script', type=str, default='ugtrack',
                        help='script name,对应 experiments/ 下的子目录名')
    parser.add_argument('--config', type=str, required=True,
                        help='yaml 配置文件名（不含扩展名），对应 experiments/<script>/ 下的文件')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='root directory to save checkpoints and logs')
    return parser.parse_args()


def main():
    args = parse_args()

    # ============================================
    # Load config
    # ============================================
    config_module = importlib.import_module('lib.config.{}.config'.format(args.script))
    cfg = config_module.cfg
    config_path = os.path.join('experiments', args.script, args.config + '.yaml')
    config_module.update_config_from_file(config_path)
    cfg.TRAIN.STAGE = 1  # force stage=1

    # ============================================
    # Settings
    # ============================================
    settings = ws_settings.Settings()
    settings.script_name = args.script
    settings.config_name = args.config
    settings.cfg_file = config_path
    settings.save_dir = os.path.abspath(args.save_dir) if args.save_dir else os.path.abspath('output')
    settings.local_rank = -1
    settings.use_wandb = False
    settings.project_path = 'train/{}/{}'.format(args.script, args.config)

    update_settings(settings, cfg)

    log_dir = os.path.join(settings.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    settings.log_file = os.path.join(log_dir, '{}-{}.log'.format(settings.script_name, settings.config_name))

    # ============================================
    # DataLoaders
    # ============================================
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # ============================================
    # Model (UWB branch only, stage=1)
    # ============================================
    net = build_ugtrack(cfg, training=True)
    net.cuda()

    # ============================================
    # Losses: L1 + BCEWithLogitsLoss
    # ============================================
    coord_loss_name = str(getattr(cfg.TRAIN, 'UWB_COORD_LOSS', 'l1')).lower()
    conf_loss_name = str(getattr(cfg.TRAIN, 'UWB_CONF_LOSS', 'bce')).lower()

    if coord_loss_name == 'l1':
        uwb_pred_loss = l1_loss
    elif coord_loss_name == 'mse':
        uwb_pred_loss = mse_loss
    else:
        raise ValueError('Unsupported UWB_COORD_LOSS: {}'.format(coord_loss_name))

    if conf_loss_name == 'bce':
        uwb_conf_loss = BCEWithLogitsLoss()
    elif conf_loss_name == 'mse':
        uwb_conf_loss = mse_loss
    else:
        raise ValueError('Unsupported UWB_CONF_LOSS: {}'.format(conf_loss_name))

    objective = {'uwb_pred': uwb_pred_loss, 'uwb_conf': uwb_conf_loss}
    loss_weight = {
        'uwb_pred': getattr(cfg.TRAIN, 'UWB_PRED_WEIGHT', 1.0),
        'uwb_conf': getattr(cfg.TRAIN, 'UWB_CONF_WEIGHT', 0.5),
    }

    # ============================================
    # Actor
    # ============================================
    actor = UGTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    # ============================================
    # Optimizer & Scheduler
    # ============================================
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    # ============================================
    # Trainer & Train
    # ============================================
    use_amp = getattr(cfg.TRAIN, 'AMP', False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=False)


if __name__ == '__main__':
    main()
