"""
UV MSE            均方误差，衡量位置预测的整体偏差（归一化坐标）
UV RMSE           均方根误差，对大误差更敏感（归一化坐标）
UV MAE (pixel)    像素级平均绝对误差，衡量实际图像空间中的位置偏差（像素）
In-box Rate       预测 UV 落在搜索区域内的比例，衡量预测是否在合理范围内
Conf MAE          置信度平均绝对误差，衡量置信度预测的数值精度
Conf RMSE         置信度均方根误差，对大偏差更敏感
Conf Pearson      置信度皮尔逊相关系数，衡量置信度预测与真值的线性相关性
Conf Spearman     置信度斯皮尔曼相关系数，衡量置信度排序一致性
All / Non-occ / Occ  分别在全体/非遮挡/遮挡样本上评估
"""
import _init_paths
import os
import glob
import importlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.train.admin import env_settings
from lib.train.data import opencv_loader
from lib.models.ugtrack import build_ugtrack

# ============================================
# Configuration — edit before running
# ============================================
split = 'test'
save_dir = 'output'                     # must match train_uwb.py --save_dir

# All UWB datasets to evaluate
# Format: (display_name, class_name, root_attr)
_datasets = [
    ('OTB100_UWB', 'OTB100UWB', 'otb100_uwb_dir'),
    ('UAV123_UWB', 'UAV123UWB', 'uav123_uwb_dir'),
    ('custom_dataset', 'CustomDataset', 'custom_dataset_dir'),
]

# Generate all 48 tracker configs: 3 encoders x 4 seq_len x 4 bce_weight
# Name format: s1_{encoder}_t{seq_len}_bce{weight_suffix}
# Weight suffix: 01=0.1, 025=0.25, 05=0.5, 10=1.0
_encoders = ['mlp', 'gru', 'tcn']
_seq_lens = [1, 3, 5, 10]
_weights = [('01', 0.1), ('025', 0.25), ('05', 0.5), ('10', 1.0)]

trackers = []
for enc in _encoders:
    for sl in _seq_lens:
        for ws, wv in _weights:
            param = 's1_{}_t{}_bce{}'.format(enc, sl, ws)
            display = '{}_{}_W{}'.format(
                enc.upper(), 'T'+str(sl), ws)
            trackers.append(dict(
                name='ugtrack',
                parameter_name=param,
                seq_len=sl,
                display_name=display,
            ))

# Global defaults (overridden by per-tracker seq_len when present)
seq_len = 10

# ============================================
# Helpers
# ============================================

def _find_latest_checkpoint(ckpt_dir):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*_ep*.pth.tar')))
    if not ckpts:
        raise FileNotFoundError('No checkpoint found in {}'.format(ckpt_dir))
    return ckpts[-1]


def compute_uv_l2(pred_uv, gt_uv):
    return torch.norm(pred_uv - gt_uv, dim=-1).cpu().numpy()


def compute_pearson(x, y):
    n = len(x)
    xm, ym = x.mean(), y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = np.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    return float(num / den) if den != 0 else 0.0


def compute_spearman(x, y):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y)[0])
    except ImportError:
        x_rank = np.argsort(np.argsort(x)).astype(np.float64)
        y_rank = np.argsort(np.argsort(y)).astype(np.float64)
        n = len(x)
        d = (x_rank - y_rank) ** 2
        return float(1 - 6 * d.sum() / (n * (n * n - 1)))


# ============================================
# Evaluate each tracker on each dataset
# ============================================
# Results keyed by dataset_name -> list of (display, m_all, m_occ, m_nonocc)
all_dataset_results = {}

for ds_display, ds_class_name, ds_root_attr in _datasets:
    print()
    print('=' * 70)
    print('  Dataset: {}, split: {}'.format(ds_display, split))
    print('=' * 70)

    # Dynamically import dataset class
    ds_module_name = ds_display.lower().replace('-', '_')
    try:
        ds_module = importlib.import_module('lib.train.dataset.{}'.format(ds_module_name))
        DSClass = getattr(ds_module, ds_class_name)
        ds_root = getattr(env_settings(), ds_root_attr)
    except (ModuleNotFoundError, AttributeError) as e:
        print('  ** SKIP: cannot load dataset {}: {}'.format(ds_display, e))
        continue

    if not os.path.isdir(ds_root):
        print('  ** SKIP: dataset directory not found: {}'.format(ds_root))
        continue

    all_results = []

    for t in trackers:
        name = t['name']
        param = t['parameter_name']
        seq_len = t.get('seq_len', seq_len)
        display = t.get('display_name', param)

        print('========== Evaluating {} (seq_len={}) on {} =========='.format(display, seq_len, ds_display))
        config_path = os.path.join('experiments', name, param + '.yaml')
        ckpt_dir = os.path.join(save_dir, 'checkpoints', 'train', name, param)
        ckpt_path = _find_latest_checkpoint(ckpt_dir)

        # Config & model
        config_module = importlib.import_module('lib.config.{}.config'.format(name))
        cfg = config_module.cfg
        config_module.update_config_from_file(config_path)
        cfg.TRAIN.STAGE = 1

        net = build_ugtrack(cfg, training=False)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        missing, unexpected = net.load_state_dict(ckpt['net'], strict=False)
        if missing:
            print('  [{}] missing_keys: {}'.format(display, missing))
        net.cuda()
        net.eval()

        # Dataset
        dataset = DSClass(root=ds_root, image_loader=opencv_loader,
                          split=split, uwb_seq_len=seq_len)
        n_seqs = dataset.get_num_sequences()

        # Inference
        coord_scale = float(cfg.DATA.SEARCH.SIZE)
        all_pred_uv, all_gt_uv = [], []
        all_conf_pred, all_gt_conf = [], []
        all_visible = []

        with torch.no_grad():
            for seq_id in range(n_seqs):
                seq_info = dataset.get_sequence_info(seq_id)
                visible = seq_info['visible'].cpu().numpy()

                for f_id in range(seq_info['bbox'].shape[0]):
                    uwb_seq = seq_info['uwb_seq'][f_id].unsqueeze(0).cuda().float()
                    gt_uv = seq_info['uwb_gt'][f_id, :2].unsqueeze(0).cuda()
                    gt_conf = seq_info['uwb_conf'][f_id].view(1, 1).cuda().float()

                    uwb_seq = (uwb_seq / coord_scale).clamp(0.0, 1.0)
                    gt_uv = (gt_uv / coord_scale).clamp(0.0, 1.0)

                    out = net(search_uwb_seq=uwb_seq, stage=1)
                    all_pred_uv.append(out['pred_uv'].cpu())
                    all_gt_uv.append(gt_uv.cpu())
                    all_conf_pred.append(out['uwb_conf_pred'].cpu())
                    all_gt_conf.append(gt_conf.cpu())
                    all_visible.append(visible[f_id])

        pred_uv = torch.cat(all_pred_uv, dim=0)
        gt_uv = torch.cat(all_gt_uv, dim=0)
        conf_pred = torch.cat(all_conf_pred, dim=0).numpy().flatten()
        gt_conf = torch.cat(all_gt_conf, dim=0).numpy().flatten()

        visible_arr = np.array(all_visible, dtype=np.int32)
        if visible_arr.max() > 1:
            visible_arr = (visible_arr == 255).astype(np.int32)

        is_visible = (visible_arr == 1)
        is_occluded = (visible_arr == 0)

        l2_norm = compute_uv_l2(pred_uv, gt_uv)             # L2 in normalized [0,1]
        l2_pixel = l2_norm * coord_scale                     # L2 in pixels
        pred_uv_np = pred_uv.numpy()
        in_box = ((pred_uv_np[:, 0] >= -0.01) & (pred_uv_np[:, 0] <= 1.01) &
                  (pred_uv_np[:, 1] >= -0.01) & (pred_uv_np[:, 1] <= 1.01))

        # Compute metrics on All / Non-occ / Occ subsets
        def compute_subset(selector):
            if selector.sum() == 0:
                return None
            e_norm = l2_norm[selector]
            e_pixel = l2_pixel[selector]
            c_pred = conf_pred[selector]
            c_gt = gt_conf[selector]
            ib = in_box[selector]
            return {
                'uv_mse': float((e_norm ** 2).mean()),
                'uv_rmse': float(np.sqrt((e_norm ** 2).mean())),
                'uv_mae_pixel': float(e_pixel.mean()),
                'inbox_rate': float(ib.mean()),
                'conf_mae': float(np.abs(c_pred - c_gt).mean()),
                'conf_rmse': float(np.sqrt(((c_pred - c_gt) ** 2).mean())),
                'conf_pearson': compute_pearson(c_pred, c_gt),
                'conf_spearman': compute_spearman(c_pred, c_gt),
            }

        m_all = compute_subset(np.ones(len(is_visible), dtype=bool))
        m_occ = compute_subset(is_occluded)
        m_nonocc = compute_subset(is_visible)

        all_results.append((display, m_all, m_occ, m_nonocc))

        # # Plot (disabled: delete eval_plots/ to remove saved figures)
        # plot_dir = os.path.join(save_dir, 'eval_plots', param)
        # os.makedirs(plot_dir, exist_ok=True)
        #
        # fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        #
        # axes[0].hist(l2_pixel, bins=50, alpha=0.7, edgecolor='black')
        # axes[0].axvline(l2_pixel.mean(), color='red', linestyle='--',
        #                 label='mean={:.2f}px'.format(l2_pixel.mean()))
        # axes[0].set_xlabel('L2 error (pixel)')
        # axes[0].set_ylabel('Count')
        # axes[0].set_title('UV Error Distribution ({})'.format(ds_display))
        # axes[0].legend()
        # axes[0].grid(True, alpha=0.3)
        #
        # axes[1].scatter(gt_conf, conf_pred, s=2, alpha=0.3)
        # axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        # axes[1].set_xlabel('Ground truth confidence')
        # axes[1].set_ylabel('Predicted confidence')
        # axes[1].set_title('Confidence Prediction ({})'.format(ds_display))
        # axes[1].set_xlim(0, 1)
        # axes[1].set_ylim(0, 1)
        # axes[1].grid(True, alpha=0.3)
        # axes[1].set_aspect('equal')
        #
        # visible_errs = l2_pixel[is_visible]
        # occluded_errs = l2_pixel[is_occluded]
        # if len(occluded_errs) > 0:
        #     axes[2].hist(visible_errs, bins=40, alpha=0.6, label='visible', color='green')
        #     axes[2].hist(occluded_errs, bins=40, alpha=0.6, label='occluded', color='red')
        #     axes[2].set_xlabel('L2 error (pixel)')
        #     axes[2].set_ylabel('Count')
        #     axes[2].set_title('Error by Visibility ({})'.format(ds_display))
        #     axes[2].legend()
        # else:
        #     axes[2].hist(l2_pixel, bins=40, alpha=0.7, edgecolor='black')
        #     axes[2].set_xlabel('L2 error (pixel)')
        #     axes[2].set_title('UV Error (no occlusion in {})'.format(ds_display))
        # axes[2].grid(True, alpha=0.3)
        #
        # plt.tight_layout()
        # plot_path = os.path.join(plot_dir, '{}_{}_uwb_eval.png'.format(ds_display.lower(), param))
        # plt.savefig(plot_path, dpi=150)
        # plt.close(fig)
        # print('Plot saved: {}'.format(plot_path))

    all_dataset_results[ds_display] = all_results

# ============================================
# Summary tables per dataset
# ============================================
def _fmt(val):
    return '{:.6f}'.format(val)

def _print_table(title, data_key, results):
    print()
    print('{}:'.format(title))
    header = '{:<22s}  {:>10s}  {:>10s}  {:>12s}  {:>9s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}'.format(
        'Tracker', 'uv_MSE', 'uv_RMSE', 'uv_MAE_px', 'In-box%', 'ConfMAE', 'ConfRMSE', 'ConfPear', 'ConfSpear')
    print(header)
    print('-' * len(header))
    for display, m_all, m_occ, m_nonocc in results:
        m = {'all': m_all, 'occ': m_occ, 'nocc': m_nonocc}[data_key]
        if m is None:
            print('{:<22s}  {:>10s}  {:>10s}  {:>12s}  {:>9s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}'.format(
                display, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'))
        else:
            print('{:<22s}  {:>10.6f}  {:>10.6f}  {:>12.6f}  {:>8.2f}%  {:>10.6f}  {:>10.6f}  {:>10.4f}  {:>10.4f}'.format(
                display, m['uv_mse'], m['uv_rmse'], m['uv_mae_pixel'],
                m['inbox_rate'] * 100,
                m['conf_mae'], m['conf_rmse'], m['conf_pearson'], m['conf_spearman']))

for ds_display in all_dataset_results:
    print()
    print('#' * 70)
    print('#  Dataset: {}'.format(ds_display))
    print('#' * 70)
    _print_table('Results - All samples ({})'.format(ds_display), 'all', all_dataset_results[ds_display])
    _print_table('Results - Non-occluded ({})'.format(ds_display), 'nocc', all_dataset_results[ds_display])
    _print_table('Results - Occluded ({})'.format(ds_display), 'occ', all_dataset_results[ds_display])
