import torch


def _get_image_shape(image):
    if image is None:
        return None
    if isinstance(image, (list, tuple)):
        image = image[0]
    if torch.is_tensor(image):
        return int(image.shape[-2]), int(image.shape[-1])
    return int(image.shape[0]), int(image.shape[1])


def transform_uwb_seq(uwb_seq, image_shape, transform_coords):
    if uwb_seq is None:
        return None

    uwb_seq = uwb_seq.clone()
    seq_shape = uwb_seq.shape
    coords = torch.stack((uwb_seq.reshape(-1, 2)[:, 1], uwb_seq.reshape(-1, 2)[:, 0]), dim=0)
    coords = transform_coords(coords, image_shape)
    xy = torch.stack((coords[1, :], coords[0, :]), dim=-1)
    return xy.reshape(seq_shape)


def transform_uwb_gt(uwb_gt, image_shape, transform_coords):
    if uwb_gt is None:
        return None

    uwb_gt = uwb_gt.clone()
    if uwb_gt.numel() < 2:
        return uwb_gt

    coords = torch.stack((uwb_gt[1:2], uwb_gt[0:1]), dim=0)
    coords = transform_coords(coords, image_shape)
    uwb_gt[0] = coords[1, 0]
    uwb_gt[1] = coords[0, 0]
    return uwb_gt


def apply_uwb_transform(transform, uwb_seq=None, uwb_gt=None, image_shape=None, new_roll=False):
    if transform is None:
        return uwb_seq, uwb_gt

    if new_roll:
        raise ValueError("apply_uwb_transform expects new_roll=False to reuse image transform randomness")

    for t in transform.transforms:
        if uwb_seq is not None:
            uwb_seq = transform_uwb_seq(
                uwb_seq, image_shape, lambda coords, shape: t.transform_coords(coords, shape, *t._rand_params))
        if uwb_gt is not None:
            uwb_gt = transform_uwb_gt(
                uwb_gt, image_shape, lambda coords, shape: t.transform_coords(coords, shape, *t._rand_params))

    return uwb_seq, uwb_gt


def apply_transform_with_uwb(transform, uwb_seq=None, uwb_gt=None, joint=True, new_roll=True, **kwargs):
    var_names = [k for k in kwargs.keys() if kwargs[k] is not None]

    if joint:
        transformed = transform(joint=True, new_roll=new_roll, **kwargs)
        if len(var_names) == 1:
            transformed = (transformed,)
        out = {k: v for k, v in zip(var_names, transformed)}
        images = kwargs.get('image')
        if isinstance(images, (list, tuple)):
            if uwb_seq is not None:
                uwb_seq = [
                    apply_uwb_transform(transform, uwb_seq=seq_i, image_shape=_get_image_shape(img_i), new_roll=False)[0]
                    for seq_i, img_i in zip(uwb_seq, images)
                ]
            if uwb_gt is not None:
                uwb_gt = [
                    apply_uwb_transform(transform, uwb_gt=gt_i, image_shape=_get_image_shape(img_i), new_roll=False)[1]
                    for gt_i, img_i in zip(uwb_gt, images)
                ]
        else:
            image_shape = _get_image_shape(images)
            uwb_seq, uwb_gt = apply_uwb_transform(
                transform, uwb_seq=uwb_seq, uwb_gt=uwb_gt, image_shape=image_shape, new_roll=False)
        return tuple(out[k] for k in var_names) + (uwb_seq, uwb_gt)

    split_inputs = []
    first_key = var_names[0]
    num_items = len(kwargs[first_key])
    for idx in range(num_items):
        split_inputs.append({k: kwargs[k][idx] for k in var_names})

    transformed_lists = {k: [] for k in var_names}
    uwb_seq_list = []
    uwb_gt_list = []

    for idx, inp in enumerate(split_inputs):
        transformed = transform(joint=True, new_roll=True, **inp)
        if len(var_names) == 1:
            transformed = (transformed,)
        for k, v in zip(var_names, transformed):
            transformed_lists[k].append(v)

        seq_i = None if uwb_seq is None else uwb_seq[idx]
        gt_i = None if uwb_gt is None else uwb_gt[idx]
        image_shape = _get_image_shape(inp.get('image'))
        seq_i, gt_i = apply_uwb_transform(
            transform, uwb_seq=seq_i, uwb_gt=gt_i, image_shape=image_shape, new_roll=False)
        if uwb_seq is not None:
            uwb_seq_list.append(seq_i)
        if uwb_gt is not None:
            uwb_gt_list.append(gt_i)

    outputs = tuple(transformed_lists[k] for k in var_names)
    outputs += (uwb_seq_list if uwb_seq is not None else None, uwb_gt_list if uwb_gt is not None else None)
    return outputs
