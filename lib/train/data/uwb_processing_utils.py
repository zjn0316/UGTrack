import torch


def as_float_tensor(x):
    if torch.is_tensor(x):
        return x.float().clone()
    return torch.as_tensor(x, dtype=torch.float32)


def transform_uwb_to_crop(uwb, box_extract: torch.Tensor, resize_factor: float, crop_sz: torch.Tensor,
                          normalize=False) -> torch.Tensor:
    uwb_out = as_float_tensor(uwb)
    if uwb_out.numel() < 2:
        return uwb_out

    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]
    if uwb_out.ndim == 1:
        uwb_xy = uwb_out[:2].reshape(1, 2)
    else:
        uwb_xy = uwb_out.reshape(-1, 2)

    uwb_center = (crop_sz - 1) / 2 + (uwb_xy - box_extract_center) * resize_factor
    if normalize:
        uwb_center = uwb_center / crop_sz[0]

    if uwb_out.ndim == 1:
        uwb_out[:2] = uwb_center[0]
    else:
        uwb_out[..., :2] = uwb_center.reshape(uwb_out[..., :2].shape)
    return uwb_out


def jittered_center_crop_uwb(uwb_seq, uwb_gt, box_extract, search_area_factor, output_sz):
    crop_sz = torch.Tensor([output_sz, output_sz])
    resize_factors = [
        float(output_sz) / torch.ceil(torch.sqrt(a[2] * a[3]) * search_area_factor)
        for a in box_extract
    ]

    uwb_seq_crop = None
    uwb_gt_crop = None

    if uwb_seq is not None:
        uwb_seq_crop = [transform_uwb_to_crop(seq, a_ex, rf, crop_sz, normalize=True).clamp(0.0, 1.0)
                        for seq, a_ex, rf in zip(uwb_seq, box_extract, resize_factors)]
    if uwb_gt is not None:
        uwb_gt_crop = []
        for gt, a_ex, rf in zip(uwb_gt, box_extract, resize_factors):
            gt_crop = transform_uwb_to_crop(gt, a_ex, rf, crop_sz, normalize=True)
            if gt_crop.numel() >= 2:
                gt_crop = gt_crop.clone()
                gt_crop[:2] = gt_crop[:2].clamp(0.0, 1.0)
            uwb_gt_crop.append(gt_crop)

    return uwb_seq_crop, uwb_gt_crop


def format_search_uwb_conf(search_uwb_conf):
    out = []
    for conf in search_uwb_conf:
        conf = as_float_tensor(conf)
        if conf.ndim == 0:
            conf = conf.unsqueeze(0)
        elif conf.ndim > 1:
            conf = conf.reshape(-1)[:1]
        out.append(conf)
    return out
