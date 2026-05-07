import _init_paths
import os
import csv
import importlib

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from PIL import Image

from lib.train.admin import env_settings
from lib.train.data import opencv_loader


# =========================================================
# config
# =========================================================
save_dir = 'output'
split = 'test'

# 要分析哪些数据集，就放哪些
dataset_names = [
    'otb100_uwb',
    'uav123_uwb',
    'custom_dataset',
]


# =========================================================
# trackerlist-style config
# =========================================================
def uwb_trackerlist(name, parameter_name, dataset_name, seq_len, run_ids=None, display_name=None):
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [{
        'name': name,
        'parameter_name': parameter_name,
        'dataset_name': dataset_name,
        'seq_len': seq_len,
        'run_id': run_id,
        'display_name': display_name if display_name is not None else parameter_name
    } for run_id in run_ids]


def build_trackers(dataset_name):
    trackers = []

    """stage1 encoder"""
    trackers.extend(uwb_trackerlist(name='ugtrack',
                                    parameter_name='stage1_encoder_mlp_seq5_confw10',
                                    dataset_name=dataset_name,
                                    seq_len=5,
                                    run_ids=None,
                                    display_name='MLP'))
    trackers.extend(uwb_trackerlist(name='ugtrack',
                                    parameter_name='stage1_encoder_gru_seq5_confw10',
                                    dataset_name=dataset_name,
                                    seq_len=5,
                                    run_ids=None,
                                    display_name='GRU'))
    trackers.extend(uwb_trackerlist(name='ugtrack',
                                    parameter_name='stage1_encoder_tcn_seq5_confw10',
                                    dataset_name=dataset_name,
                                    seq_len=5,
                                    run_ids=None,
                                    display_name='TCN'))
    return trackers


# =========================================================
# dataset registry
# =========================================================
_DATASET_INFO = {
    'otb100_uwb': {
        'display_name': 'OTB100_UWB',
        'module_name': 'otb100_uwb',
        'class_name': 'OTB100UWB',
        'root_attr': 'otb100_uwb_dir',
    },
    'uav123_uwb': {
        'display_name': 'UAV123_UWB',
        'module_name': 'uav123_uwb',
        'class_name': 'UAV123UWB',
        'root_attr': 'uav123_uwb_dir',
    },
    'custom_dataset': {
        'display_name': 'CUSTOM_DATASET',
        'module_name': 'custom_dataset',
        'class_name': 'CustomDataset',
        'root_attr': 'custom_dataset_dir',
    },
}


# =========================================================
# helpers
# =========================================================
def get_uwb_dataset(dataset_name, split='test', seq_len=5):
    dataset_key = dataset_name.lower()
    if dataset_key not in _DATASET_INFO:
        raise ValueError('Unknown dataset: {}. Available: {}'.format(
            dataset_name, list(_DATASET_INFO.keys()))
        )

    info = _DATASET_INFO[dataset_key]
    ds_module = importlib.import_module('lib.train.dataset.{}'.format(info['module_name']))
    ds_class = getattr(ds_module, info['class_name'])
    ds_root = getattr(env_settings(), info['root_attr'])

    dataset = ds_class(
        root=ds_root,
        image_loader=opencv_loader,
        split=split,
        uwb_seq_len=seq_len
    )
    return dataset, info['display_name']


def _results_dir(base_save_dir, dataset_name, tracker_name, parameter_name):
    # 注意：这里必须用小写 dataset_name，而不是 display_name
    return os.path.join(base_save_dir, 'test', 'uwb_results', dataset_name, tracker_name, parameter_name)


def _safe_load_txt(path):
    if not os.path.isfile(path):
        raise FileNotFoundError('Missing result file: {}'.format(path))
    arr = np.loadtxt(path, delimiter='\t')
    if arr.ndim == 0:
        arr = np.array([arr])
    return arr


def _compute_uv_l2(pred_uv_norm, gt_uv_norm):
    return np.linalg.norm(pred_uv_norm - gt_uv_norm, axis=1)


def _compute_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    xm, ym = x.mean(), y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = np.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    return float(num / den) if den != 0 else 0.0


def _compute_spearman(x, y):
    try:
        from scipy.stats import spearmanr
        val = spearmanr(x, y)[0]
        return float(val) if val == val else 0.0
    except ImportError:
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)
        if n <= 1:
            return 0.0
        xr = np.argsort(np.argsort(x)).astype(np.float64)
        yr = np.argsort(np.argsort(y)).astype(np.float64)
        d = (xr - yr) ** 2
        return float(1 - 6 * d.sum() / (n * (n * n - 1)))


def _compute_auc_from_errors(err, max_thr=0.05, num=101):
    if len(err) == 0:
        return 0.0
    thrs = np.linspace(0.0, max_thr, num=num)
    curve = [(err <= t).mean() for t in thrs]
    auc = np.trapz(curve, thrs) / max_thr
    return float(auc)


def _get_image_size(seq_path):
    """Return (width, height) of the first frame image in the sequence."""
    first_img = sorted([f for f in os.listdir(seq_path)
                        if f.endswith(('.jpg', '.jpeg', '.png'))])[0]
    img = Image.open(os.path.join(seq_path, first_img))
    return img.size  # (width, height)


def _compute_subset_metrics(selector, l2_norm, l2_pixel, conf_pred, gt_conf, pred_uv_img, img_w, img_h):
    if selector.sum() == 0:
        return None

    e_norm = l2_norm[selector]
    e_pixel = l2_pixel[selector]
    c_pred = conf_pred[selector]
    c_gt = gt_conf[selector]
    pred_sub = pred_uv_img[selector]

    # Check whether prediction falls within the valid image coordinate range
    in_range = (
        (pred_sub[:, 0] >= 0.0) & (pred_sub[:, 0] <= img_w) &
        (pred_sub[:, 1] >= 0.0) & (pred_sub[:, 1] <= img_h)
    )

    return {
        'uv_pred_auc': _compute_auc_from_errors(e_norm, max_thr=0.05, num=101),
        'uv_l2_mean': float(e_norm.mean()),
        'uv_mse': float((e_norm ** 2).mean()),
        'uv_rmse': float(np.sqrt((e_norm ** 2).mean())),
        'uv_mae_pixel': float(e_pixel.mean()),
        'inrange_rate': float(in_range.mean()),
        'conf_mae': float(np.abs(c_pred - c_gt).mean()),
        'conf_rmse': float(np.sqrt(((c_pred - c_gt) ** 2).mean())),
        'conf_pearson': _compute_pearson(c_pred, c_gt),
        'conf_spearman': _compute_spearman(c_pred, c_gt),
    }


def _print_table(title, subset_key, results):
    print()
    print(title)
    header = (
        f"{'Tracker':<8s}  "
        f"{'AUC':>8s}  "
        f"{'L2mean':>10s}  "
        f"{'uv_MAE_px':>12s}  "
        f"{'In-range%':>10s}  "
        f"{'ConfMAE':>10s}  "
        f"{'ConfRMSE':>10s}  "
        f"{'ConfPear':>10s}  "
        f"{'ConfSpear':>10s}"
    )
    print(header)
    print('-' * len(header))

    for row in results:
        m = row[subset_key]
        if m is None:
            print(f"{row['tracker']:<8s}  {'N/A':>8s}  {'N/A':>10s}  {'N/A':>12s}  {'N/A':>9s}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>10s}")
        else:
            print(
                f"{row['tracker']:<8s}  "
                f"{m['uv_pred_auc']:>8.4f}  "
                f"{m['uv_l2_mean']:>10.6f}  "
                f"{m['uv_mae_pixel']:>12.6f}  "
                f"{m['inrange_rate'] * 100:>9.2f}%  "
                f"{m['conf_mae']:>10.6f}  "
                f"{m['conf_rmse']:>10.6f}  "
                f"{m['conf_pearson']:>10.4f}  "
                f"{m['conf_spearman']:>10.4f}"
            )


def print_uwb_results(trackers, dataset, dataset_name, dataset_display_name, split='test', save_dir='output'):
    """Evaluate UWB prediction results.

    test_uwb.py now saves predictions in image-pixel space (using center-relative
    normalization matching training, then denormalizing back to pixels).
    GT is also in image pixel space — they can be compared directly.
    """
    dataset_results = []
    csv_rows = []

    for t in trackers:
        tracker_name = t['name']
        parameter_name = t['parameter_name']
        display_name = t['display_name']

        result_dir = _results_dir(save_dir, dataset_name, tracker_name, parameter_name)
        print('>>> Evaluating {} from {}'.format(display_name, result_dir))

        all_pred_uv_img = []
        all_gt_uv_img = []
        all_conf_pred = []
        all_gt_conf = []
        all_visible = []
        all_img_w = []
        all_img_h = []

        for seq_id in range(dataset.get_num_sequences()):
            seq_name = dataset.sequence_list[seq_id]
            seq_info = dataset.get_sequence_info(seq_id)
            seq_path = dataset._get_sequence_path(seq_id)

            pred_uv_path = os.path.join(result_dir, '{}_pred_uv.txt'.format(seq_name))
            conf_path = os.path.join(result_dir, '{}_conf.txt'.format(seq_name))

            pred_uv_pixel = _safe_load_txt(pred_uv_path)
            conf_pred = _safe_load_txt(conf_path).reshape(-1)

            gt_uv_pixel = seq_info['uwb_gt'][:, :2].cpu().numpy()
            gt_conf = seq_info['uwb_conf'].cpu().numpy().reshape(-1)
            visible = seq_info['visible'].cpu().numpy().reshape(-1)

            if pred_uv_pixel.ndim == 1:
                pred_uv_pixel = pred_uv_pixel.reshape(-1, 2)

            if pred_uv_pixel.shape[0] != gt_uv_pixel.shape[0]:
                raise ValueError(
                    'Frame count mismatch in {}: pred {} vs gt {}'.format(
                        seq_name, pred_uv_pixel.shape[0], gt_uv_pixel.shape[0]
                    )
                )

            img_w, img_h = _get_image_size(seq_path)
            n_frames = pred_uv_pixel.shape[0]

            all_pred_uv_img.append(pred_uv_pixel)
            all_gt_uv_img.append(gt_uv_pixel)
            all_conf_pred.append(conf_pred)
            all_gt_conf.append(gt_conf)
            all_visible.append(visible)
            all_img_w.append(np.full(n_frames, img_w, dtype=np.float32))
            all_img_h.append(np.full(n_frames, img_h, dtype=np.float32))

        pred_uv_img = np.concatenate(all_pred_uv_img, axis=0)
        gt_uv_img = np.concatenate(all_gt_uv_img, axis=0)
        img_w_arr = np.concatenate(all_img_w, axis=0)
        img_h_arr = np.concatenate(all_img_h, axis=0)
        conf_pred = np.concatenate(all_conf_pred, axis=0)
        gt_conf = np.concatenate(all_gt_conf, axis=0)
        visible_arr = np.concatenate(all_visible, axis=0).astype(np.int32)

        if visible_arr.max() > 1:
            visible_arr = (visible_arr == 255).astype(np.int32)

        is_visible = (visible_arr == 1)
        is_occluded = (visible_arr == 0)

        # Normalized L2 (per-axis by image dimensions, for AUC)
        l2_norm = np.sqrt(
            ((pred_uv_img[:, 0] - gt_uv_img[:, 0]) / img_w_arr) ** 2 +
            ((pred_uv_img[:, 1] - gt_uv_img[:, 1]) / img_h_arr) ** 2
        )
        # Pixel L2 (for uv_MAE_px)
        l2_pixel = np.sqrt(
            (pred_uv_img[:, 0] - gt_uv_img[:, 0]) ** 2 +
            (pred_uv_img[:, 1] - gt_uv_img[:, 1]) ** 2
        )

        m_all = _compute_subset_metrics(
            np.ones(len(l2_norm), dtype=bool),
            l2_norm, l2_pixel, conf_pred, gt_conf,
            pred_uv_img, img_w_arr[0], img_h_arr[0]
        )
        m_nocc = _compute_subset_metrics(
            is_visible,
            l2_norm, l2_pixel, conf_pred, gt_conf,
            pred_uv_img, img_w_arr[0], img_h_arr[0]
        )
        m_occ = _compute_subset_metrics(
            is_occluded,
            l2_norm, l2_pixel, conf_pred, gt_conf,
            pred_uv_img, img_w_arr[0], img_h_arr[0]
        )

        dataset_results.append({
            'tracker': display_name,
            'all': m_all,
            'nocc': m_nocc,
            'occ': m_occ,
        })

        for subset_name, subset_metrics in [('all', m_all), ('nocc', m_nocc), ('occ', m_occ)]:
            row = {
                'dataset': dataset_display_name,
                'tracker': display_name,
                'subset': subset_name
            }
            if subset_metrics is None:
                row.update({
                    'uv_pred_auc': None,
                    'uv_l2_mean': None,
                    'uv_mse': None,
                    'uv_rmse': None,
                    'uv_mae_pixel': None,
                    'inrange_rate': None,
                    'conf_mae': None,
                    'conf_rmse': None,
                    'conf_pearson': None,
                    'conf_spearman': None,
                })
            else:
                row.update(subset_metrics)
            csv_rows.append(row)

    dataset_results.sort(
        key=lambda x: (
            -(x['all']['uv_pred_auc'] if x['all'] is not None else -1e9),
            x['all']['uv_mae_pixel'] if x['all'] is not None else 1e9,
            x['all']['conf_mae'] if x['all'] is not None else 1e9,
        )
    )

    _print_table('Results - All ({})'.format(dataset_display_name), 'all', dataset_results)
    _print_table('Results - Non-occ ({})'.format(dataset_display_name), 'nocc', dataset_results)
    _print_table('Results - Occ ({})'.format(dataset_display_name), 'occ', dataset_results)

    print()
    print('Ranking on {} (by All/AUC desc, then All/uv_mae_pixel, then All/conf_mae):'.format(dataset_display_name))
    for i, row in enumerate(dataset_results, 1):
        m = row['all']
        if m is None:
            print('{}. {} | N/A'.format(i, row['tracker']))
        else:
            print('{}. {} | uv_pred_auc={:.4f}, uv_mae_pixel={:.4f}, conf_mae={:.4f}, conf_spearman={:.4f}'.format(
                i, row['tracker'], m['uv_pred_auc'], m['uv_mae_pixel'], m['conf_mae'], m['conf_spearman'])
            )

    return dataset_results, csv_rows


if __name__ == '__main__':
    os.makedirs(os.path.join(save_dir, 'analysis'), exist_ok=True)

    all_csv_rows = []
    overall_summary = {'MLP': [], 'GRU': [], 'TCN': []}

    for dataset_name in dataset_names:
        trackers = build_trackers(dataset_name)
        dataset, dataset_display_name = get_uwb_dataset(dataset_name, split=split, seq_len=trackers[0]['seq_len'])

        dataset_results, csv_rows = print_uwb_results(
            trackers=trackers,
            dataset=dataset,
            dataset_name=dataset_name,                 # 小写，用于路径
            dataset_display_name=dataset_display_name, # 大写，用于显示
            split=split,
            save_dir=save_dir
        )

        all_csv_rows.extend(csv_rows)

        for row in dataset_results:
            if row['all'] is not None:
                overall_summary[row['tracker']].append(row['all'])

    print()
    print('=' * 80)
    print('Overall ranking (mean over datasets, by AUC desc, then uv_mae_pixel, then conf_mae)')
    print('=' * 80)

    overall_rows = []
    for tracker_name, metrics_list in overall_summary.items():
        if not metrics_list:
            continue
        mean_auc = float(np.mean([m['uv_pred_auc'] for m in metrics_list]))
        mean_uv = float(np.mean([m['uv_mae_pixel'] for m in metrics_list]))
        mean_conf = float(np.mean([m['conf_mae'] for m in metrics_list]))
        mean_spear = float(np.mean([m['conf_spearman'] for m in metrics_list]))
        overall_rows.append((tracker_name, mean_auc, mean_uv, mean_conf, mean_spear))

    overall_rows.sort(key=lambda x: (-x[1], x[2], x[3]))

    for i, (tracker_name, mean_auc, mean_uv, mean_conf, mean_spear) in enumerate(overall_rows, 1):
        print('{}. {} | mean_uv_pred_auc={:.4f}, mean_uv_mae_pixel={:.4f}, mean_conf_mae={:.4f}, mean_conf_spearman={:.4f}'.format(
            i, tracker_name, mean_auc, mean_uv, mean_conf, mean_spear
        ))

    csv_path = os.path.join(save_dir, 'analysis', 'analysis_uwb_results_all_{}.csv'.format(split))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'dataset', 'tracker', 'subset',
                'uv_pred_auc', 'uv_l2_mean', 'uv_mse', 'uv_rmse', 'uv_mae_pixel',
                'inrange_rate', 'conf_mae', 'conf_rmse', 'conf_pearson', 'conf_spearman'
            ]
        )
        writer.writeheader()
        writer.writerows(all_csv_rows)

    print()
    print('CSV saved to: {}'.format(csv_path))