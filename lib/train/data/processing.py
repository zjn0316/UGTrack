import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """数据预处理基类。

    Processing 用于在数据进入网络前，对数据集返回的样本进行统一预处理。
    例如：围绕目标裁剪搜索区域、执行数据增强等。
    """
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        参数:
            transform - 图像默认变换。当 template_transform 或 search_transform 为 None 时使用。
            template_transform - 模板图像的变换流程。若为 None，则回退到 transform。
            search_transform - 搜索图像的变换流程。若为 None，则回退到 transform。
            joint_transform - 对模板与搜索图像同时应用的联合变换。
                              例如可将模板与搜索图像同步转换为灰度图。
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ 用于 LittleBoy 训练的数据预处理类。图像的处理流程如下：
    首先，对目标的边界框（bounding box）进行扰动（加噪声）。
    然后，以扰动后的目标中心为中心，
    从原图中裁剪出一个正方形区域（称为搜索区域，search region），该区域面积为扰动后目标框面积的 search_area_factor^2 倍。
    对目标框进行扰动的目的是避免模型学习到“目标总在搜索区域中心”的偏置。
    最后，将搜索区域缩放到 output_sz 指定的固定尺寸。
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        参数说明：
            search_area_factor - 搜索区域相对于目标尺寸的缩放因子。
            output_sz - 搜索区域缩放到的输出尺寸（整数，始终为正方形）。
            center_jitter_factor - 一个 dict，表示在裁剪搜索区域前，对目标中心进行扰动的幅度。具体扰动方式见 _get_jittered_box。
            scale_jitter_factor - 一个 dict，表示在裁剪搜索区域前，对目标尺寸进行扰动的幅度。具体扰动方式见 _get_jittered_box。
            mode - 'pair' 或 'sequence'，若为 'sequence'，输出会多一维帧数。
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """对输入边界框进行随机扰动。

        参数:
            box - 输入边界框。
            mode - 'template' 或 'search'，用于区分模板分支与搜索分支。
        返回:
            torch.Tensor - 扰动后的边界框。
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        参数:
            data - 输入数据，通常应包含以下字段：
                   'template_images'、'search_images'、'template_anno'、'search_anno'。
        返回:
            TensorDict - 处理后的数据字典。
        """
        # 应用联合变换
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "在 pair 模式下，模板帧与搜索帧数量必须为 1"

            # 扰动bbox
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 避免bbox过小
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # 以扰动后的目标为中心裁剪区域，生成注意力掩码
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[s],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            # 应用独立transform
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 检查注意力掩码是否全为 1（无效）
            # 注意：data[s + '_att'] 的类型是 tuple，其中每个 ele 是 torch.Tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
                
                # 检查下采样后是否全为 1（无效）
                feat_size = self.output_sz[s] // 16  # 16 是 backbone 的步长
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # 若使用 copy-and-paste 增强但未提供 mask，则补零 mask
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # 整理输出张量形状
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
