import random
import torch
import torch.utils.data

from lib.utils import TensorDict
from .sampler import TrackingSampler, no_processing
import numpy as np

class UWBTrackingSampler(TrackingSampler):
    """
    在官方 TrackingSampler 基础上做最小扩展：
    - 保留原有 seq / frame id 采样逻辑
    - 仅在打包 TensorDict 时，额外加入 UWB 相关字段
    - 不在 sampler 内做归一化、坐标变换、裁剪映射，这些留给 processing
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing,
                 frame_sample_mode='causal', train_cls=False, pos_prob=0.5):
        super().__init__(
            datasets=datasets,
            p_datasets=p_datasets,
            samples_per_epoch=samples_per_epoch,
            max_gap=max_gap,
            num_search_frames=num_search_frames,
            num_template_frames=num_template_frames,
            processing=processing,
            frame_sample_mode=frame_sample_mode,
            train_cls=train_cls,
            pos_prob=pos_prob
        )

    #============================
    # 区别于官方原版的内容：
    # 重写 getitem，只扩充 UWB 字段
    #============================
    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False

        while not valid:
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            is_video_dataset = dataset.is_video_sequence()
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(
                            visible,
                            num_ids=1,
                            min_id=self.num_template_frames - 1,
                            max_id=len(visible) - self.num_search_frames
                        )
                        prev_frame_ids = self._sample_visible_ids(
                            visible,
                            num_ids=self.num_template_frames - 1,
                            min_id=base_frame_id[0] - self.max_gap - gap_increase,
                            max_id=base_frame_id[0]
                        )
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue

                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(
                            visible,
                            min_id=template_frame_ids[0] + 1,
                            max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                            num_ids=self.num_search_frames
                        )
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)

                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(
                        visible, seq_info_dict["valid"]
                    )
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames

            try:
                template_frames, template_anno, meta_obj_train = dataset.get_frames(
                    seq_id, template_frame_ids, seq_info_dict
                )
                search_frames, search_anno, meta_obj_test = dataset.get_frames(
                    seq_id, search_frame_ids, seq_info_dict
                )

                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else \
                    [torch.zeros((H, W))] * self.num_template_frames
                search_masks = search_anno['mask'] if 'mask' in search_anno else \
                    [torch.zeros((H, W))] * self.num_search_frames

                #============================
                # 区别于官方原版的内容：
                # 统一把 UWB 相关字段也放进 TensorDict
                #============================
                data = TensorDict({
                    'template_images': template_frames,
                    'template_anno': template_anno['bbox'],
                    'template_masks': template_masks,
                    
                    'search_images': search_frames,
                    'search_anno': search_anno['bbox'],
                    'search_masks': search_masks,
                    'search_uwb_seq': search_anno['uwb_seq'] ,
                    'search_uwb_gt': search_anno['uwb_gt'] ,
                    'search_uwb_conf': search_anno['uwb_conf'],

                    'dataset': dataset.get_name(),
                    'test_class': meta_obj_test.get('object_class_name'),

                    # 额外保留元信息，方便 processing/debug
                    'seq_id': seq_id,
                    'template_frame_ids': template_frame_ids,
                    'search_frame_ids': search_frame_ids,
                })

                # 预处理
                data = self.processing(data)

                valid = data['valid']

            except Exception as e:
                valid = False

        return data
