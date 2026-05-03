import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class CustomDataset(BaseDataset):
    """ CustomDataset (测试用)

    数据集结构:
    CustomDataset/
    ├── train/
    ├── val/
    └── test/
        ├── <sequence_name>/
        │   ├── 00000001.jpg
        │   ├── groundtruth.txt
        │   ├── valid.txt
        │   ├── occlusion.txt
        │   ├── uwb_obs.txt
        │   ├── uwb_gt.txt
        │   └── uwb_conf.txt
        └── list.txt
    """

    def __init__(self, split='test'):
        """
        args:
            split - 'train', 'val', or 'test'
        """
        super().__init__()
        self.base_path = self.env_settings.custom_dataset_path
        self.split = split
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = [
            '{base_path}/{split}/{sequence_path}/{frame:0{nz}}.{ext}'.format(
                base_path=self.base_path,
                split=self.split,
                sequence_path=sequence_path,
                frame=frame_num,
                nz=nz,
                ext=ext
            )
            for frame_num in range(start_frame + init_omit, end_frame + 1)
        ]

        anno_path = '{}/{}/{}'.format(self.base_path, self.split, sequence_info['anno_path'])

        ground_truth_rect = load_text(
            str(anno_path),
            delimiter=(',', None),
            dtype=np.float64,
            backend='numpy'
        )

        init_data = {
            0: {
                "bbox": ground_truth_rect[init_omit, :],
                "uwb_obs_path": '{}/{}/{}/uwb_obs.txt'.format(self.base_path, self.split, sequence_path),
                "uwb_gt_path": '{}/{}/{}/uwb_gt.txt'.format(self.base_path, self.split, sequence_path),
                "uwb_conf_path": '{}/{}/{}/uwb_conf.txt'.format(self.base_path, self.split, sequence_path),
                "valid_path": '{}/{}/{}/valid.txt'.format(self.base_path, self.split, sequence_path),
                "occlusion_path": '{}/{}/{}/occlusion.txt'.format(self.base_path, self.split, sequence_path),
            }
        }

        return Sequence(
            sequence_info['name'],
            frames,
            'custom_dataset',
            ground_truth_rect[init_omit:, :],
            init_data=init_data,
            object_class=sequence_info.get('object_class', 'unknown')
        )

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        """从 list.txt 读取序列列表并构建序列信息"""
        split_dir = os.path.join(self.base_path, self.split)
        list_file = os.path.join(split_dir, 'list.txt')

        if os.path.exists(list_file):
            with open(list_file, 'r', encoding='utf-8') as f:
                seq_names = [line.strip() for line in f.readlines() if line.strip()]
        else:
            seq_names = [
                d for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]
            seq_names.sort()

        sequence_info_list = []

        for seq_name in seq_names:
            seq_dir = os.path.join(split_dir, seq_name)

            if not os.path.isdir(seq_dir):
                continue

            img_files = [
                f for f in os.listdir(seq_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ]
            if not img_files:
                continue

            img_numbers = []
            valid_img_files = []

            for img in img_files:
                stem = os.path.splitext(img)[0]
                try:
                    num = int(stem)
                    img_numbers.append(num)
                    valid_img_files.append(img)
                except Exception:
                    pass

            if not img_numbers:
                continue

            start_frame = min(img_numbers)
            end_frame = max(img_numbers)

            sample_img = sorted(valid_img_files)[0]
            stem, ext = os.path.splitext(sample_img)
            nz = len(stem)
            ext = ext.lstrip('.')

            seq_info = {
                "name": seq_name,
                "path": seq_name,
                "startFrame": start_frame,
                "endFrame": end_frame,
                "nz": nz,
                "ext": ext,
                "anno_path": "{}/groundtruth.txt".format(seq_name),
                "object_class": "unknown"
            }
            sequence_info_list.append(seq_info)

        return sequence_info_list