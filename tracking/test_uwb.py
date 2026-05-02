import argparse
import glob
import importlib
import os
import sys
import time

import numpy as np
import torch

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import _init_paths
from lib.train.admin import env_settings as train_env_settings
from lib.train.data import opencv_loader
from lib.train.dataset import OTB100UWB
from lib.models.ugtrack import build_ugtrack


_dataset_map = {
    'otb100_uwb': OTB100UWB,
}


def _create_dataset(name, split, seq_len):
    cls = _dataset_map.get(name.lower())
    if cls is None:
        raise ValueError('Unknown UWB dataset: {} (available: {})'.format(name, list(_dataset_map.keys())))
    return cls(
        root=train_env_settings().otb100_uwb_dir,
        image_loader=opencv_loader,
        split=split,
        uwb_seq_len=seq_len,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Run UWB evaluation on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--dataset_name', type=str, default='otb100_uwb', help='Name of the UWB dataset')
    parser.add_argument('--split', type=str, default='test', help='dataset split (default: test)')
    parser.add_argument('--seq_len', type=int, default=10, help='UWB sequence length')
    return parser.parse_args()


def _find_latest_checkpoint(ckpt_dir):
    """Find the checkpoint with the highest epoch number."""
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*_ep*.pth.tar')))
    if not ckpts:
        raise FileNotFoundError('No checkpoint found in {}'.format(ckpt_dir))
    return ckpts[-1]


def main():
    args = parse_args()

    # -----------------------------------------------------------
    # Derive paths
    # -----------------------------------------------------------
    save_dir = 'output'
    config_path = os.path.join('experiments', args.tracker_name, args.tracker_param + '.yaml')
    ckpt_dir = os.path.join(save_dir, 'checkpoints', 'train', args.tracker_name, args.tracker_param)
    ckpt_path = _find_latest_checkpoint(ckpt_dir)

    # -----------------------------------------------------------
    # Config & model
    # -----------------------------------------------------------
    config_module = importlib.import_module('lib.config.{}.config'.format(args.tracker_name))
    cfg = config_module.cfg
    config_module.update_config_from_file(config_path)
    cfg.TRAIN.STAGE = 1

    coord_scale = float(cfg.DATA.SEARCH.SIZE)  # normalize to [0,1] like training

    net = build_ugtrack(cfg, training=False)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = net.load_state_dict(ckpt['net'], strict=False)
    print('Checkpoint: {}'.format(ckpt_path))
    print('  missing_keys: {}'.format(missing))
    print('  unexpected_keys: {}'.format(unexpected))
    net.cuda()
    net.eval()

    # -----------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------
    dataset = _create_dataset(args.dataset_name, args.split, args.seq_len)
    n_seqs = dataset.get_num_sequences()
    print('Dataset: {} [{}], {} sequences'.format(args.dataset_name, args.split, n_seqs))

    # -----------------------------------------------------------
    # Output: {save_dir}/test/tracking_results/{tracker_name}/{tracker_param}/
    # -----------------------------------------------------------
    out_dir = os.path.abspath(os.path.join(save_dir, 'test', 'uwb_results', args.tracker_name, args.tracker_param))
    os.makedirs(out_dir, exist_ok=True)
    print('Output: {}'.format(out_dir))

    # -----------------------------------------------------------
    # Per-sequence inference
    # -----------------------------------------------------------
    all_total_time = 0.0
    all_num_frames = 0

    for seq_id in range(n_seqs):
        seq_name = dataset.sequence_list[seq_id]
        seq_info = dataset.get_sequence_info(seq_id)
        n_frames = seq_info['bbox'].shape[0]

        pred_uv_list = []
        conf_pred_list = []
        frame_time_list = []

        with torch.no_grad():
            for f_id in range(n_frames):
                search_uwb_seq = seq_info['uwb_seq'][f_id].unsqueeze(0).cuda().float()
                search_uwb_seq = (search_uwb_seq / coord_scale).clamp(0.0, 1.0)

                start = time.time()
                out = net(search_uwb_seq=search_uwb_seq, stage=1)
                elapsed = time.time() - start

                pred_uv_pixel = out['pred_uv'].squeeze(0).cpu().numpy() * coord_scale
                pred_uv_list.append(pred_uv_pixel)
                conf_pred_list.append(out['uwb_conf_pred'].squeeze(0).cpu().numpy())
                frame_time_list.append(elapsed)

        pred_uv_arr = np.array(pred_uv_list)
        conf_arr = np.array(conf_pred_list)
        time_arr = np.array(frame_time_list)

        np.savetxt(os.path.join(out_dir, '{}_pred_uv.txt'.format(seq_name)),
                   pred_uv_arr, delimiter='\t', fmt='%.6f')
        np.savetxt(os.path.join(out_dir, '{}_conf.txt'.format(seq_name)),
                   conf_arr, delimiter='\t', fmt='%.6f')
        np.savetxt(os.path.join(out_dir, '{}_time.txt'.format(seq_name)),
                   time_arr, delimiter='\t', fmt='%.6f')

        total_time = sum(frame_time_list)
        all_total_time += total_time
        all_num_frames += n_frames
        print('  {:20s}  {:4d} frames  total={:.2f}s  fps={:.1f}'.format(
            seq_name, n_frames, total_time, n_frames / max(total_time, 1e-12)))

    # -----------------------------------------------------------
    # Summary
    # -----------------------------------------------------------
    avg_fps = all_num_frames / max(all_total_time, 1e-12)
    summary = (
        'UGTrack Stage-1 Results\n'
        'Tracker: {}\n'
        'Config: {}\n'
        'Checkpoint: {}\n'
        'Split: {}\n'
        'Sequences: {}\n'
        'Total frames: {}\n'
        'Total time: {:.2f}s\n'
        'Average FPS: {:.1f}\n'
    ).format(args.tracker_name, args.tracker_param, ckpt_path, args.split,
             n_seqs, all_num_frames, all_total_time, avg_fps)

    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print('\n' + summary)
    print('Results saved to: {}'.format(out_dir))


if __name__ == '__main__':
    main()
