import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAV123UWBDataset(BaseDataset):
    """ UAV123_UWB dataset (测试用，默认仍以视觉跟踪评测为主)

    这是按统一格式整理后的 UAV123_UWB 版本，包含 UWB 标签文件。
    测试时默认仍以视觉 groundtruth 为主，同时将 UWB 文件路径写入 init_data。

    数据集结构:
    /home/zjn/data/UAV123_UWB/
    ├── train/
    ├── val/
    └── test/
        ├── <sequence_name>/
        │   ├── 00000001.jpg  (8位数字编号)
        │   ├── groundtruth.txt
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
        self.base_path = self.env_settings.uav123_uwb_path
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

        # UAV123_UWB 使用 8 位数字编号
        frames = ['{base_path}/{split}/{sequence_path}/{frame:0{nz}}.{ext}'.format(
            base_path=self.base_path,
            split=self.split,
            sequence_path=sequence_path,
            frame=frame_num,
            nz=nz,
            ext=ext
        ) for frame_num in range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}/{}'.format(self.base_path, self.split, sequence_info['anno_path'])

        # 读取标注文件
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        init_data = {
            0: {
                "bbox": ground_truth_rect[init_omit, :],
                "uwb_obs_path": '{}/{}/{}/uwb_obs.txt'.format(self.base_path, self.split, sequence_path),
                "uwb_gt_path": '{}/{}/{}/uwb_gt.txt'.format(self.base_path, self.split, sequence_path),
                "uwb_conf_path": '{}/{}/{}/uwb_conf.txt'.format(self.base_path, self.split, sequence_path),
            }
        }

        return Sequence(sequence_info['name'], frames, 'uav123_uwb', ground_truth_rect[init_omit:, :],
                        init_data=init_data, object_class=sequence_info.get('object_class', 'unknown'))

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        """从 list.txt 读取序列列表并构建序列信息"""
        import os

        split_dir = os.path.join(self.base_path, self.split)
        list_file = os.path.join(split_dir, 'list.txt')

        if os.path.exists(list_file):
            # 从 list.txt 读取
            with open(list_file, 'r') as f:
                seq_names = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # 直接扫描目录
            seq_names = [d for d in os.listdir(split_dir)
                         if os.path.isdir(os.path.join(split_dir, d))]

        sequence_info_list = []
        for seq_name in seq_names:
            seq_dir = os.path.join(split_dir, seq_name)

            # 检查图像格式和数量
            img_files = [f for f in os.listdir(seq_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if not img_files:
                continue

            # 获取第一张和最后一张图片的帧号
            img_numbers = []
            for img in img_files:
                try:
                    num = int(img.split('.')[0])
                    img_numbers.append(num)
                except Exception:
                    pass

            if not img_numbers:
                continue

            start_frame = min(img_numbers)
            end_frame = max(img_numbers)

            # 判断编号位数
            sample_img = img_files[0]
            nz = len(sample_img.split('.')[0])
            ext = sample_img.split('.')[-1]

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