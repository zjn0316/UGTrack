import torch


def rect_to_rel(bb, sz_norm=None):
    """将标准矩形参数化的边界框 [x, y, w, h]
    转换为相对参数化 [cx/sw, cy/sh, log(w), log(h)]，其中 [cx, cy] 为中心坐标。
    args:
        bb  -  N x 4 的边界框张量。
        sz_norm  -  [N] x 2 的 [sw, sh] 张量（可选）。若未提供，则默认 sw=w、sh=h。
    """

    c = bb[...,:2] + 0.5 * bb[...,2:]
    if sz_norm is None:
        c_rel = c / bb[...,2:]
    else:
        c_rel = c / sz_norm
    sz_rel = torch.log(bb[...,2:])
    return torch.cat((c_rel, sz_rel), dim=-1)


def rel_to_rect(bb, sz_norm=None):
    """执行 rect_to_rel 的逆变换，详见上方说明。"""

    sz = torch.exp(bb[...,2:])
    if sz_norm is None:
        c = bb[...,:2] * sz
    else:
        c = bb[...,:2] * sz_norm
    tl = c - 0.5 * sz
    return torch.cat((tl, sz), dim=-1)


def masks_to_bboxes(mask, fmt='c'):

    """将掩码张量转换为一个或多个边界框。
    注意：该函数相对较新，请确认其行为与说明一致。/Andreas
    :param mask: 掩码张量，形状 = (..., H, W)
    :param fmt: 边界框格式。'c' => "中心点 + 尺寸"，即 (x_center, y_center, width, height)
                             't' => "左上角 + 尺寸"，即 (x_left, y_top, width, height)
                             'v' => "顶点坐标"，即 (x_left, y_top, x_right, y_bottom)
    :return: 包含一批边界框的张量，形状 = (..., 4)
    """
    batch_shape = mask.shape[:-2]
    mask = mask.reshape((-1, *mask.shape[-2:]))
    bboxes = []

    for m in mask:
        mx = m.sum(dim=-2).nonzero()
        my = m.sum(dim=-1).nonzero()
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]
        bboxes.append(bb)

    bboxes = torch.tensor(bboxes, dtype=torch.float32, device=mask.device)
    bboxes = bboxes.reshape(batch_shape + (4,))

    if fmt == 'v':
        return bboxes

    x1 = bboxes[..., :2]
    s = bboxes[..., 2:] - x1 + 1

    if fmt == 'c':
        return torch.cat((x1 + 0.5 * s, s), dim=-1)
    elif fmt == 't':
        return torch.cat((x1, s), dim=-1)

    raise ValueError("Undefined bounding box layout '%s'" % fmt)


def masks_to_bboxes_multi(mask, ids, fmt='c'):
    assert mask.dim() == 2
    bboxes = []

    for id in ids:
        mx = (mask == id).sum(dim=-2).nonzero()
        my = (mask == id).float().sum(dim=-1).nonzero()
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]

        bb = torch.tensor(bb, dtype=torch.float32, device=mask.device)

        x1 = bb[:2]
        s = bb[2:] - x1 + 1

        if fmt == 'v':
            pass
        elif fmt == 'c':
            bb = torch.cat((x1 + 0.5 * s, s), dim=-1)
        elif fmt == 't':
            bb = torch.cat((x1, s), dim=-1)
        else:
            raise ValueError("Undefined bounding box layout '%s'" % fmt)
        bboxes.append(bb)

    return bboxes
