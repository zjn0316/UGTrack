import os
import sys
import glob
import time
import argparse
import importlib

import numpy as np
import torch

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import _init_paths
from lib.train.admin import env_settings
from lib.train.data import opencv_loader
from lib.models.ugtrack import build_ugtrack


# =========================================================
# dataset registry
# =========================================================
_DATASET_INFO = {
    'otb100_uwb': {
        'module_name': 'otb100_uwb',
        'class_name': 'OTB100UWB',
        'root_attr': 'otb100_uwb_dir',
    },
    'uav123_uwb': {
        'module_name': 'uav123_uwb',
        'class_name': 'UAV123UWB',
        'root_attr': 'uav123_uwb_dir',
    },
    'custom_dataset': {
        'module_name': 'custom_dataset',
        'class_name': 'CustomDataset',
        'root_attr': 'custom_dataset_dir',
    },
}


# =========================================================
# helpers
# =========================================================
def _find_latest_checkpoint(ckpt_dir):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*_ep*.pth.tar')))
    if not ckpts:
        raise FileNotFoundError('No checkpoint found in {}'.format(ckpt_dir))
    return ckpts[-1]


def _load_stage1_model(tracker_name, tracker_param, save_dir='output'):
    config_path = os.path.join('experiments', tracker_name, tracker_param + '.yaml')
    ckpt_dir = os.path.join(save_dir, 'checkpoints', 'train', tracker_name, tracker_param)
    ckpt_path = _find_latest_checkpoint(ckpt_dir)

    config_module = importlib.import_module('lib.config.{}.config'.format(tracker_name))
    config_module = importlib.reload(config_module)
    cfg = config_module.cfg
    config_module.update_config_from_file(config_path)
    cfg.TRAIN.STAGE = 1

    net = build_ugtrack(cfg, training=False)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = net.load_state_dict(ckpt['net'], strict=False)

    net.cuda()
    net.eval()

    return cfg, net, ckpt_path, missing, unexpected


def _create_dataset(dataset_name, split, seq_len):
    dataset_key = dataset_name.lower()
    if dataset_key not in _DATASET_INFO:
        raise ValueError('Unknown dataset: {}. Available: {}'.format(
            dataset_name, list(_DATASET_INFO.keys()))
        )

    info = _DATASET_INFO[dataset_key]
    ds_module = importlib.import_module('lib.train.dataset.{}'.format(info['module_name']))
    ds_class = getattr(ds_module, info['class_name'])
    ds_root = getattr(env_settings(), info['root_attr'])

    return ds_class(
        root=ds_root,
        image_loader=opencv_loader,
        split=split,
        uwb_seq_len=seq_len
    )


def _normalize_uwb_seq(search_uwb_seq, coord_scale):
    """Center-relative normalization matching training preprocessing.

    Training uses: norm = ((crop_sz-1)/2 + (uwb - bbox_center) * resize) / crop_sz
    where bbox_center ≈ last UWB observation (both on the object).

    We approximate: norm = (uwb - last_obs) / coord_scale + 0.5
    This centers the last observation at 0.5, matching the training distribution.
    """
    last_obs = search_uwb_seq[:, -1:, :]  # (B, 1, 2) — anchor point
    return ((search_uwb_seq - last_obs) / coord_scale + 0.5).clamp(0.0, 1.0)


def _run_one_sequence(net, seq_info, coord_scale):
    n_frames = seq_info['bbox'].shape[0]

    pred_uv_list = []
    conf_pred_list = []
    frame_time_list = []

    with torch.no_grad():
        for f_id in range(n_frames):
            search_uwb_seq = seq_info['uwb_seq'][f_id].unsqueeze(0).cuda().float()
            last_obs = search_uwb_seq[:, -1, :].clone()  # (1, 2) — denorm anchor
            search_uwb_seq_norm = _normalize_uwb_seq(search_uwb_seq, coord_scale)

            start = time.time()
            out = net(search_uwb_seq=search_uwb_seq_norm, stage=1)
            elapsed = time.time() - start

            # Denormalize: invert the center-relative normalization
            pred_uv_norm = out['pred_uv'].squeeze(0).cpu().numpy()  # (2,)
            pred_uv_pixel = (pred_uv_norm - 0.5) * coord_scale + last_obs.squeeze(0).cpu().numpy()
            conf_pred = out['uwb_conf_pred'].squeeze(0).cpu().numpy()

            pred_uv_list.append(pred_uv_pixel)
            conf_pred_list.append(conf_pred)
            frame_time_list.append(elapsed)

    return (
        np.array(pred_uv_list, dtype=np.float32),
        np.array(conf_pred_list, dtype=np.float32),
        np.array(frame_time_list, dtype=np.float32),
    )


def run_tracker(tracker_name,
                tracker_param,
                dataset_name='otb100_uwb',
                split='test',
                seq_len=None,
                sequence=None,
                save_dir='output'):
    cfg, net, ckpt_path, missing, unexpected = _load_stage1_model(
        tracker_name, tracker_param, save_dir=save_dir
    )

    if seq_len is None:
        seq_len = int(cfg.DATA.UWB.SEQ_LEN)

    coord_scale = float(cfg.DATA.SEARCH.SIZE)
    dataset = _create_dataset(dataset_name, split, seq_len)

    if sequence is not None:
        if isinstance(sequence, int):
            seq_ids = [sequence]
        else:
            seq_ids = [dataset.sequence_list.index(sequence)]
    else:
        seq_ids = list(range(dataset.get_num_sequences()))

    out_dir = os.path.abspath(os.path.join(
        save_dir, 'test', 'uwb_results', dataset_name, tracker_name, tracker_param
    ))
    os.makedirs(out_dir, exist_ok=True)

    print('----- Test UWB: {} -----'.format(tracker_param))
    print('Checkpoint: {}'.format(ckpt_path))
    print('  missing_keys: {}'.format(missing))
    print('  unexpected_keys: {}'.format(unexpected))
    print('Dataset: {} [{}], {} sequences'.format(dataset_name, split, len(seq_ids)))
    print('Output: {}'.format(out_dir))

    all_total_time = 0.0
    all_num_frames = 0

    for seq_id in seq_ids:
        seq_name = dataset.sequence_list[seq_id]
        seq_info = dataset.get_sequence_info(seq_id)

        pred_uv_arr, conf_arr, time_arr = _run_one_sequence(net, seq_info, coord_scale)

        np.savetxt(
            os.path.join(out_dir, '{}_pred_uv.txt'.format(seq_name)),
            pred_uv_arr, delimiter='\t', fmt='%.6f'
        )
        np.savetxt(
            os.path.join(out_dir, '{}_conf.txt'.format(seq_name)),
            conf_arr, delimiter='\t', fmt='%.6f'
        )
        np.savetxt(
            os.path.join(out_dir, '{}_time.txt'.format(seq_name)),
            time_arr, delimiter='\t', fmt='%.6f'
        )

        total_time = float(time_arr.sum())
        num_frames = int(seq_info['bbox'].shape[0])

        all_total_time += total_time
        all_num_frames += num_frames

        print('  {:20s}  {:4d} frames  total={:.2f}s  fps={:.1f}'.format(
            seq_name, num_frames, total_time, num_frames / max(total_time, 1e-12))
        )

    avg_fps = all_num_frames / max(all_total_time, 1e-12)

    summary = (
        'UGTrack Stage-1 UWB Results\n'
        'Tracker: {}\n'
        'Config: {}\n'
        'Checkpoint: {}\n'
        'Dataset: {}\n'
        'Split: {}\n'
        'Seq len: {}\n'
        'Sequences: {}\n'
        'Total frames: {}\n'
        'Total time: {:.2f}s\n'
        'Average FPS: {:.1f}\n'
    ).format(
        tracker_name,
        tracker_param,
        ckpt_path,
        dataset_name,
        split,
        seq_len,
        len(seq_ids),
        all_num_frames,
        all_total_time,
        avg_fps
    )

    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    print('\n' + summary)
    print('Results saved to: {}'.format(out_dir))


def main():
    parser = argparse.ArgumentParser(description='Run UWB evaluation on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--dataset_name', type=str, default='otb100_uwb',
                        help='Dataset name: otb100_uwb / uav123_uwb / custom_dataset')
    parser.add_argument('--split', type=str, default='test', help='Dataset split: train/val/test')
    parser.add_argument('--seq_len', type=int, default=None, help='Override UWB sequence length')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence index or sequence name')
    parser.add_argument('--save_dir', type=str, default='output', help='Save directory')
    args = parser.parse_args()

    try:
        seq_name = int(args.sequence) if args.sequence is not None else None
    except Exception:
        seq_name = args.sequence

    run_tracker(
        tracker_name=args.tracker_name,
        tracker_param=args.tracker_param,
        dataset_name=args.dataset_name,
        split=args.split,
        seq_len=args.seq_len,
        sequence=seq_name,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()