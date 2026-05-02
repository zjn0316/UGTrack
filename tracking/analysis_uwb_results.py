import _init_paths
import os
import glob
import importlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.train.admin import env_settings
from lib.train.data import opencv_loader
from lib.train.dataset import OTB100UWB
from lib.models.ugtrack import build_ugtrack

# ============================================
# Configuration — edit before running
# ============================================
dataset_name = 'otb100_uwb'
split = 'test'
seq_len = 10
save_dir = 'output'                     # must match train_uwb.py --save_dir

trackers = []
# UGTrack
# trackers.append(dict(name='ugtrack', parameter_name='s1_best_t10_bce05', display_name='UGTrack_best'))
trackers.append(dict(name='ugtrack', parameter_name='s1_tcn_t10_bce05', display_name='UGTrack_TCN'))
# trackers.append(dict(name='ugtrack', parameter_name='s1_gru_t10_bce05', display_name='UGTrack_GRU'))
# trackers.append(dict(name='ugtrack', parameter_name='s1_mlp_t10_bce05', display_name='UGTrack_MLP'))

# ============================================
# Helpers
# ============================================

def _find_latest_checkpoint(ckpt_dir):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*_ep*.pth.tar')))
    if not ckpts:
        raise FileNotFoundError('No checkpoint found in {}'.format(ckpt_dir))
    return ckpts[-1]


def compute_uv_error(pred_uv, gt_uv):
    return torch.norm(pred_uv - gt_uv, dim=-1).cpu().numpy()


def compute_uv_pred_auc(errors, threshold_max=0.50, step=0.01):
    thresholds = np.arange(step, threshold_max + step, step)
    success_rates = np.array([(errors < t).mean() for t in thresholds])
    norm_auc = np.trapz(success_rates, thresholds) / threshold_max
    return thresholds, success_rates, norm_auc


def _roc_auc(labels, scores):
    order = np.argsort(scores)
    labels_sorted = labels[order]
    pos_count = labels_sorted.sum()
    neg_count = len(labels_sorted) - pos_count
    if pos_count == 0 or neg_count == 0:
        return float('nan')
    rank_sum = (labels_sorted == 1).nonzero()[0].sum()
    return (rank_sum - pos_count * (pos_count - 1) / 2) / (pos_count * neg_count)


def compute_conf_auc(conf_pred, errors, error_thresh=0.05):
    labels = (errors < error_thresh).astype(np.float32)
    conf_scores = conf_pred.flatten()
    if len(np.unique(labels)) < 2:
        return float('nan')
    return _roc_auc(labels, conf_scores)


def compute_occlusion_auc(conf_pred, visible_flags):
    labels = visible_flags.astype(np.float32)
    conf_scores = conf_pred.flatten()
    if len(np.unique(labels)) < 2:
        return float('nan')
    return _roc_auc(labels, conf_scores)


def compute_losses(pred_uv, gt_uv, conf_logit, gt_conf):
    from torch.nn.functional import l1_loss, binary_cross_entropy_with_logits
    pred_loss = l1_loss(pred_uv, gt_uv[..., :2]).item()
    conf_loss = binary_cross_entropy_with_logits(conf_logit, gt_conf).item()
    return pred_loss, conf_loss, pred_loss + conf_loss


# ============================================
# Evaluate each tracker
# ============================================
print('Dataset: {}, split: {}, seq_len: {}'.format(dataset_name, split, seq_len))
print()

all_metrics = []

for t in trackers:
    name = t['name']
    param = t['parameter_name']
    display = t.get('display_name', param)

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
    dataset = OTB100UWB(root=env_settings().otb100_uwb_dir, image_loader=opencv_loader,
                        split=split, uwb_seq_len=seq_len)
    n_seqs = dataset.get_num_sequences()

    # Inference
    coord_scale = float(cfg.DATA.SEARCH.SIZE)
    all_pred_uv, all_gt_uv = [], []
    all_conf_pred, all_conf_logit = [], []
    all_gt_conf, all_visible = [], []

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
                all_conf_logit.append(out['uwb_conf_logit'].cpu())
                all_gt_conf.append(gt_conf.cpu())
                all_visible.append(visible[f_id])

    pred_uv = torch.cat(all_pred_uv, dim=0)
    gt_uv = torch.cat(all_gt_uv, dim=0)
    conf_pred = torch.cat(all_conf_pred, dim=0)
    conf_logit = torch.cat(all_conf_logit, dim=0)
    gt_conf = torch.cat(all_gt_conf, dim=0)
    visible_arr = np.array(all_visible)

    # Metrics
    pred_loss, conf_loss, total_loss = compute_losses(pred_uv, gt_uv, conf_logit, gt_conf)
    errors = compute_uv_error(pred_uv, gt_uv)
    _, _, uv_auc = compute_uv_pred_auc(errors)
    conf_auc = compute_conf_auc(conf_pred, errors, error_thresh=0.05)
    occ_auc = compute_occlusion_auc(conf_pred, visible_arr)

    all_metrics.append((display, total_loss, pred_loss, conf_loss, uv_auc, conf_auc, occ_auc))

    # Plot
    plot_dir = os.path.join(save_dir, 'eval_plots', param)
    os.makedirs(plot_dir, exist_ok=True)

    thresholds, success_rates, _ = compute_uv_pred_auc(errors)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(thresholds, success_rates, linewidth=2)
    axes[0].fill_between(thresholds, success_rates, alpha=0.2)
    axes[0].axvline(0.05, color='gray', linestyle='--', alpha=0.5, label='thresh=0.05')
    axes[0].set_title('UV Prediction AUC = {:.4f}'.format(uv_auc))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(errors.mean(), color='red', linestyle='--',
                    label='mean={:.4f}'.format(errors.mean()))
    axes[1].set_title('UV Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    visible_errs = errors[visible_arr == 1]
    occluded_errs = errors[visible_arr == 0]
    if len(occluded_errs) > 0:
        axes[2].hist(visible_errs, bins=40, alpha=0.6, label='visible', color='green')
        axes[2].hist(occluded_errs, bins=40, alpha=0.6, label='occluded', color='red')
    else:
        axes[2].hist(errors, bins=40, alpha=0.7, edgecolor='black')
    axes[2].set_title('Error by Visibility')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, '{}_uwb_eval.png'.format(param))
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print('Plot saved: {}'.format(plot_path))

# ============================================
# Summary table
# ============================================
print()
print('Results:')
print('{:<24s}  {:>8s}  {:>8s}  {:>8s}  {:>10s}  {:>10s}  {:>12s}'.format(
    'Tracker', 'total', 'pred', 'conf', 'uv_auc', 'conf_auc', 'occ_auc'))
print('-' * 90)
for m in all_metrics:
    print('{:<24s}  {:8.5f}  {:8.5f}  {:8.5f}  {:10.4f}  {:10.4f}  {:12.4f}'.format(*m))
