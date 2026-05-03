import os
import os.path
import random
from collections import OrderedDict

import numpy as np
import pandas
import torch

from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class CustomDataset(BaseVideoDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train', seq_ids=None,
                 data_fraction=None, uwb_seq_len=5):

        root = env_settings().custom_dataset_dir if root is None else root
        super().__init__('CustomDataset', root, image_loader)

        self.split = split
        self.split_root = os.path.join(self.root, self.split)
        self.uwb_seq_len = uwb_seq_len

        self.sequence_list = self._get_sequence_list()

        if seq_ids is None:
            seq_ids = list(range(len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(
                self.sequence_list,
                int(len(self.sequence_list) * data_fraction)
            )

        self._sequence_info_cache = {}
        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'custom_dataset'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        list_file = os.path.join(self.split_root, 'list.txt')

        if os.path.isfile(list_file):
            sequence_list = pandas.read_csv(list_file, header=None).squeeze("columns").values.tolist()
        else:
            sequence_list = [
                d for d in sorted(os.listdir(self.split_root))
                if os.path.isdir(os.path.join(self.split_root, d))
            ]

        return sequence_list

    def _load_meta_info(self):
        return {s: self._build_meta_info(s) for s in self.sequence_list}

    def _build_meta_info(self, seq_name):
        return OrderedDict({
            'object_class_name': seq_name,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.split_root, self.sequence_list[seq_id])

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.jpg'.format(frame_id + 1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        return obj_meta['object_class_name']

    def _read_occlusion_anno(self, seq_path):
        occlusion_file = os.path.join(seq_path, "occlusion.txt")
        occlusion = pandas.read_csv(
            occlusion_file, header=None, dtype=np.float32,
            na_filter=False, low_memory=False
        ).values.reshape(-1)
        return torch.tensor(occlusion, dtype=torch.float32)

    def _read_valid_anno(self, seq_path):
        valid_file = os.path.join(seq_path, "valid.txt")
        valid = pandas.read_csv(
            valid_file, header=None, dtype=np.float32,
            na_filter=False, low_memory=False
        ).values.reshape(-1)
        return torch.tensor(valid, dtype=torch.float32)

    def _read_target_visible(self, seq_path):
        occlusion = self._read_occlusion_anno(seq_path)
        occlusion = (occlusion > 0).byte()
        target_visible = ~occlusion
        visible_ratio = 1.0 - occlusion.float()
        return target_visible, visible_ratio

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(
            bb_anno_file, delimiter=',', header=None, dtype=np.float32,
            na_filter=False, low_memory=False
        ).values
        return torch.tensor(gt, dtype=torch.float32)

    def _read_uwb_gt_anno(self, seq_path):
        uwb_gt_file = os.path.join(seq_path, "uwb_gt.txt")
        uwb_gt = pandas.read_csv(
            uwb_gt_file, delimiter=',', header=None, dtype=np.float32,
            na_filter=False, low_memory=False
        ).values
        return torch.tensor(uwb_gt, dtype=torch.float32)

    def _read_uwb_obs_anno(self, seq_path):
        uwb_obs_file = os.path.join(seq_path, "uwb_obs.txt")
        uwb_obs = pandas.read_csv(
            uwb_obs_file, delimiter=',', header=None, dtype=np.float32,
            na_filter=False, low_memory=False
        ).values
        return torch.tensor(uwb_obs, dtype=torch.float32)

    def _read_uwb_conf_anno(self, seq_path):
        uwb_conf_file = os.path.join(seq_path, "uwb_conf.txt")
        uwb_conf = pandas.read_csv(
            uwb_conf_file, header=None, dtype=np.float32,
            na_filter=False, low_memory=False
        ).values.reshape(-1)
        return torch.tensor(uwb_conf, dtype=torch.float32)

    def _build_uwb_seq(self, uwb_obs):
        uv = uwb_obs[:, :2]
        uwb_seq = []

        for frame_id in range(uv.shape[0]):
            seq_list = []
            for i in range(self.uwb_seq_len):
                hist_id = frame_id - self.uwb_seq_len + 1 + i
                if hist_id < 0:
                    hist_id = 0
                seq_list.append(uv[hist_id])

            uwb_seq.append(torch.stack(seq_list))

        return torch.stack(uwb_seq)

    def _build_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)

        bbox = self._read_bb_anno(seq_path)
        valid_txt = self._read_valid_anno(seq_path)
        visible, visible_ratio = self._read_target_visible(seq_path)

        # 以 valid.txt 为主，再与 bbox 合法性做一次与
        bbox_valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid = (valid_txt > 0) & bbox_valid

        uwb_gt = self._read_uwb_gt_anno(seq_path)
        uwb_obs = self._read_uwb_obs_anno(seq_path)
        uwb_seq = self._build_uwb_seq(uwb_obs)
        uwb_conf = self._read_uwb_conf_anno(seq_path)

        return {
            'bbox': bbox,                  # [N, 4]
            'valid': valid,                # [N]
            'visible': visible,            # [N]
            'visible_ratio': visible_ratio,# [N]
            'uwb_gt': uwb_gt,              # [N, 2]
            'uwb_obs': uwb_obs,            # [N, 2]
            'uwb_seq': uwb_seq,            # [N, T, 2]
            'uwb_conf': uwb_conf,          # [N]
        }

    def get_sequence_info(self, seq_id):
        seq_info = self._sequence_info_cache.get(seq_id)
        if seq_info is None:
            seq_info = self._build_sequence_info(seq_id)
            self._sequence_info_cache[seq_id] = seq_info
        return seq_info

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if value.ndim == 1:
                anno_frames[key] = [value[f_id].clone() for f_id in frame_ids]
            else:
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta