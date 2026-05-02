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


class OTB100UWB(BaseVideoDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train', seq_ids=None,
                 data_fraction=None, uwb_seq_len=5):

        root = env_settings().otb100_uwb_dir if root is None else root
        super().__init__('OTB100UWB', root, image_loader)

        self.split = split
        self.split_root = os.path.join(self.root, self.split)
        self.uwb_seq_len = uwb_seq_len

        self.sequence_list = self._get_sequence_list()

        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self._sequence_info_cache = {}
        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'otb100_uwb'

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
            sequence_list = [d for d in sorted(os.listdir(self.split_root))
                             if os.path.isdir(os.path.join(self.split_root, d))]

        return sequence_list

    def _load_meta_info(self):
        sequence_meta_info = {s: self._build_meta_info(s) for s in self.sequence_list}
        return sequence_meta_info

    def _build_meta_info(self, seq_name):
        object_meta = OrderedDict({'object_class_name': seq_name,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return object_meta

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
        return os.path.join(seq_path, '{:08}.jpg'.format(frame_id + 1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        return obj_meta['object_class_name']
    
    def _read_occlusion_anno(self, seq_path):
        occlusion_file = os.path.join(seq_path, "occlusion.txt")
        occlusion = pandas.read_csv(occlusion_file, header=None, dtype=np.float32, na_filter=False, low_memory=False).values.reshape(-1)
        return torch.tensor(occlusion, dtype=torch.float32)
    
    def _read_target_visible(self, seq_path):
        occlusion = self._read_occlusion_anno(seq_path)

        occlusion = (occlusion > 0).byte()
        target_visible = ~occlusion
        visible_ratio = 1.0 - occlusion.float()

        return target_visible, visible_ratio

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_uwb_gt_anno(self, seq_path):
        uwb_gt_file = os.path.join(seq_path, "uwb_gt.txt")
        uwb_gt = pandas.read_csv(uwb_gt_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(uwb_gt, dtype=torch.float32)

    def _read_uwb_obs_anno(self, seq_path):
        uwb_obs_file = os.path.join(seq_path, "uwb_obs.txt")
        uwb_obs = pandas.read_csv(uwb_obs_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(uwb_obs, dtype=torch.float32)


    def _build_uwb_seq(self, seq_path, uwb):
        uv = uwb[:, :2]
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

    def _read_uwb_conf_anno(self, seq_path):
        uwb_conf_file = os.path.join(seq_path, "uwb_conf.txt")
        uwb_conf = pandas.read_csv(uwb_conf_file, header=None, dtype=np.float32, na_filter=False, low_memory=False).values.reshape(-1)
        return torch.tensor(uwb_conf, dtype=torch.float32)
    
    def _build_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible, visible_ratio = self._read_target_visible(seq_path)

        uwb_gt = self._read_uwb_gt_anno(seq_path)
        uwb_obs = self._read_uwb_obs_anno(seq_path)
        uwb_seq = self._build_uwb_seq(seq_path, uwb_obs)

        uwb_conf = self._read_uwb_conf_anno(seq_path)


        return {
            'bbox': bbox,
            'valid': valid,
            'visible': visible,
            'visible_ratio': visible_ratio,
            'uwb_gt': uwb_gt,
            'uwb_obs': uwb_obs,
            'uwb_seq': uwb_seq,
            'uwb_conf': uwb_conf,
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
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta

"""
调用教程 (Usage Tutorial):

OTB100UWB 数据集类主要由训练框架中的采样器 (Sampler) 调用。其核心数据流如下：
1. 采样器通过 `get_sequence_info` 获取整个视频序列的所有标注张量。
2. 采样器根据选定的帧索引 (frame_ids)，调用 `get_frames` 提取对应帧的图像和标注切片。

示例代码 (Example Usage):
--------------------------------------------------
from lib.train.dataset.otb100_uwb import OTB100UWB
from lib.train.data import jpeg4py_loader

# 1. 实例化数据集
# root: 数据集根目录，包含 train/val/test 文件夹
dataset = OTB100UWB(root='data/OTB100_UWB', split='train', uwb_seq_len=5)

# 2. 获取序列索引及其基本信息
seq_id = 0
seq_name = dataset.sequence_list[seq_id]
print(f"Loading sequence: {seq_name}")

# 3. 获取完整序列标注 (通常由 Sampler 内部缓存)
seq_info = dataset.get_sequence_info(seq_id)
# 返回值字段包括: bbox[N,4], valid[N], visible[N], uwb_gt[N,5], uwb_obs[N,5], uwb_seq[N,5,2], uwb_conf[N]

# 4. 提取特定帧的数据 (模拟一次训练采样)
frame_ids = [0, 50, 100]  # 请求第1帧、第51帧和第101帧
frame_list, anno_frames, obj_meta = dataset.get_frames(seq_id, frame_ids, anno=seq_info)

# 数据结构说明:
# - frame_list: 包含 3 张图像数组的列表 [img, img, img]
# - anno_frames: 标注字典，每个 key 对应长度为 3 的列表。
#   例如: anno_frames['bbox'][0] 是第1帧的 [x, y, w, h] Tensor。
#         anno_frames['uwb_seq'][0] 是第1帧对应的 [5, 2] 历史 UWB 序列。
#         anno_frames['uwb_conf'][0] 是第1帧的 uwb_conf 比例标量。
# - obj_meta: 包含序列元数据，如 {'object_class_name': 'Panda', ...}
--------------------------------------------------
"""
