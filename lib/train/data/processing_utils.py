import torch
import math
import cv2 as cv
import torch.nn.functional as F
import numpy as np

'''基于原始测试实现修改
将 cv.BORDER_REPLICATE 替换为 cv.BORDER_CONSTANT
新增变量 att_mask，用于后续计算注意力和位置编码'''


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """以 target_bb 为中心裁剪正方形区域，其面积为 target_bb 面积的 search_area_factor^2 倍

    args:
        im - cv 图像
        target_bb - 目标框 [x, y, w, h]
        search_area_factor - 裁剪尺寸与目标尺寸的比例
        output_sz - (float) 提取区域缩放后的尺寸（始终为正方形）。若为 None，则不进行缩放。

    returns:
        cv image - 提取后的裁剪图像
        float - 裁剪区域被缩放到 output_size 时的缩放因子
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # 裁剪图像
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # 裁剪目标区域
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # 边界填充
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # 处理注意力掩码
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded


def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """将框坐标从原图坐标系转换到裁剪图坐标系
    args:
        box_in - 需要进行坐标变换的框
        box_extract - 提取图像裁剪区域时所依据的框
        resize_factor - 原图尺度与裁剪图尺度之间的比例
        crop_sz - 裁剪图尺寸

    returns:
        torch.Tensor - box_in 变换后的坐标
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """对 frames 中每一帧，以 box_extract 为中心提取正方形区域，面积为 box_extract 面积的
    search_area_factor^2 倍。提取后将裁剪图缩放到 output_sz，并将 box_gt 的坐标
    转换到裁剪图坐标系中

    args:
        frames - 帧列表
        box_extract - 与 frames 等长的框列表，按该框进行裁剪
        box_gt - 与 frames 等长的框列表，这些框的坐标将从图像坐标转换到裁剪图坐标
        search_area_factor - 提取区域面积为 box_extract 面积的 search_area_factor^2 倍
        output_sz - 提取后裁剪图缩放到的尺寸

    returns:
        list - 裁剪图像列表
        list - box_gt 在裁剪图坐标系中的位置
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: ndarray 元组 (128,128,3), att_mask: ndarray 元组 (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # 计算目标框在裁剪图中的位置
    '''注意：这里使用的是归一化坐标'''
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors
    return frames_crop, box_crop, att_mask, masks_crop


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """将框坐标从原图坐标系转换到裁剪图坐标系
    args:
        box - 需要进行坐标变换的框
        crop_box - 原图中定义裁剪区域的边界框
        crop_sz - 裁剪图尺寸

    returns:
        torch.Tensor - 变换后的框坐标
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out

