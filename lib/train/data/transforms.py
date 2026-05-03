import random
import numpy as np
import math
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf


class Transform:
    """
    一组变换操作，常用于数据增强等场景。
    构造函数参数：
        transforms: 任意数量的变换操作，需继承自 TransformBase 类。它们会按给定顺序依次应用。

    Transform 对象可以对图像、边界框和分割掩码进行联合变换。
    只需以关键字参数方式调用该对象即可（所有参数均为可选）。

    以下参数为待变换的输入，可以是单个实例，也可以是实例列表：
        image  -  图像
        coords  -  2xN 维的二维图像坐标张量 [y, x]
        bbox  -  边界框，格式为 [x, y, w, h]
        mask  -  离散类别的分割掩码

    调用 transform 对象时可额外传入如下参数：
        joint [Bool]  -  若为 True，则对列表中的所有 image/coords/bbox/mask 使用同一组随机参数进行联合变换。
                         否则，每组 (images, coords, bbox, mask) 独立使用不同的随机参数进行变换。默认值：True。
        new_roll [Bool]  -  若为 False，则不重新采样随机参数，直接复用上一次的结果。默认值：True。

    具体用法可参考 DiMPProcessing 类。
    """

    def __init__(self, *transforms):
        """
        初始化 Transform 类。
        参数：
            *transforms: 变换操作序列，依次应用。
        """
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = transforms[0]
        self.transforms = transforms
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['joint', 'new_roll']
        self._valid_all = self._valid_inputs + self._valid_args

    def __call__(self, **inputs):
        """
        调用该对象，对输入数据进行变换。
        参数：
            支持 image, coords, bbox, mask, att 等关键字参数。
            joint/new_roll 控制变换方式。
        返回：
            变换后的数据，顺序与输入一致。
        """
        # 检查输入参数是否合法
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        for v in inputs.keys():
            if v not in self._valid_all:
                raise ValueError('Incorrect input \"{}\" to transform. Only supports inputs {} and arguments {}.'.format(v, self._valid_inputs, self._valid_args))

        joint_mode = inputs.get('joint', True)
        new_roll = inputs.get('new_roll', True)

        # 单独变换
        if not joint_mode:
            out = zip(*[self(**inp) for inp in self._split_inputs(inputs)])
            return tuple(list(o) for o in out)

        out = {k: v for k, v in inputs.items() if k in self._valid_inputs}

        for t in self.transforms:
            out = t(**out, joint=joint_mode, new_roll=new_roll)
        if len(var_names) == 1:
            return out[var_names[0]]
        # Make sure order is correct
        return tuple(out[v] for v in var_names)

    def _split_inputs(self, inputs):
        """
        将输入按元素拆分成多个字典，便于分别处理每组数据。
        参数：inputs - 输入数据字典
        返回：
            拆分后的输入字典列表
        """
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        split_inputs = [{k: v for k, v in zip(var_names, vals)} for vals in zip(*[inputs[vn] for vn in var_names])]
        for arg_name, arg_val in filter(lambda it: it[0]!='joint' and it[0] in self._valid_args, inputs.items()):
            if isinstance(arg_val, list):
                for inp, av in zip(split_inputs, arg_val):
                    inp[arg_name] = av
            else:
                for inp in split_inputs:
                    inp[arg_name] = arg_val
        return split_inputs

    def __repr__(self):
        """
        返回该变换对象的字符串表示，便于打印和调试。
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class TransformBase:
    """Base class for transformation objects. See the Transform class for details."""
    def __init__(self):
        """
        初始化 TransformBase 类。
        设置支持的输入类型和参数。
        """
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['new_roll']
        self._valid_all = self._valid_inputs + self._valid_args
        self._rand_params = None

    def __call__(self, **inputs):
        """
        调用该对象，对输入数据进行变换。
        参数：
            支持 image, coords, bbox, mask, att 等关键字参数。
            new_roll 控制是否重新采样随机参数。
        返回：
            变换后的数据字典。
        """
        # 拆分输入
        input_vars = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        input_args = {k: v for k, v in inputs.items() if k in self._valid_args}

        # 采样随机参数
        if input_args.get('new_roll', True):
            rand_params = self.roll()
            if rand_params is None:
                rand_params = ()
            elif not isinstance(rand_params, tuple):
                rand_params = (rand_params,)
            self._rand_params = rand_params

        outputs = dict()
        for var_name, var in input_vars.items():
            if var is not None:
                transform_func = getattr(self, 'transform_' + var_name)
                if var_name in ['coords', 'bbox']:
                    params = (self._get_image_size(input_vars),) + self._rand_params
                else:
                    params = self._rand_params
                if isinstance(var, (list, tuple)):
                    outputs[var_name] = [transform_func(x, *params) for x in var]
                else:
                    outputs[var_name] = transform_func(var, *params)
        return outputs

    def _get_image_size(self, inputs):
        """
        获取输入图像或掩码的尺寸。
        参数：inputs - 输入数据字典
        返回：(H, W) 或 None
        """
        im = None
        for var_name in ['image', 'mask']:
            if inputs.get(var_name) is not None:
                im = inputs[var_name]
                break
        if im is None:
            return None
        if isinstance(im, (list, tuple)):
            im = im[0]
        if isinstance(im, np.ndarray):
            return im.shape[:2]
        if torch.is_tensor(im):
            return (im.shape[-2], im.shape[-1])
        raise Exception('Unknown image type')

    def roll(self):
        """
        采样本次变换所需的随机参数。
        返回：
            随机参数（可为 None/单值/元组）
        """
        return None

    def transform_image(self, image, *rand_params):
        """
        对图像进行变换。必须是确定性的。
        参数：image - 输入图像，rand_params - 随机参数
        返回：变换后的图像
        """
        return image

    def transform_coords(self, coords, image_shape, *rand_params):
        """
        对坐标进行变换。必须是确定性的。
        参数：coords - 坐标，image_shape - 图像尺寸，rand_params - 随机参数
        返回：变换后的坐标
        """
        return coords

    def transform_bbox(self, bbox, image_shape, *rand_params):
        """
        对边界框进行变换，假设格式为 [x, y, w, h]。
        参数：bbox - 边界框，image_shape - 图像尺寸，rand_params - 随机参数
        返回：变换后的边界框
        """
        # 检查是否重载了 transform_coords
        if self.transform_coords.__code__ == TransformBase.transform_coords.__code__:
            return bbox

        coord = bbox.clone().view(-1,2).t().flip(0)

        x1 = coord[1, 0]
        x2 = coord[1, 0] + coord[1, 1]

        y1 = coord[0, 0]
        y2 = coord[0, 0] + coord[0, 1]

        coord_all = torch.tensor([[y1, y1, y2, y2], [x1, x2, x2, x1]])

        coord_transf = self.transform_coords(coord_all, image_shape, *rand_params).flip(0)
        tl = torch.min(coord_transf, dim=1)[0]
        sz = torch.max(coord_transf, dim=1)[0] - tl
        bbox_out = torch.cat((tl, sz), dim=-1).reshape(bbox.shape)
        return bbox_out

    def transform_mask(self, mask, *rand_params):
        """
        对掩码进行变换。必须是确定性的。
        参数：mask - 掩码，rand_params - 随机参数
        返回：变换后的掩码
        """
        return mask

    def transform_att(self, att, *rand_params):
        """
        对注意力掩码进行变换。
        参数：att - 注意力掩码，rand_params - 随机参数
        返回：变换后的注意力掩码
        """
        return att


class ToTensor(TransformBase):
    """Convert to a Tensor"""

    def transform_image(self, image):
        # handle numpy array
        if image.ndim == 2:
            image = image[:, :, None]

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(image, torch.ByteTensor):
            return image.float().div(255)
        else:
            return image

    def transfrom_mask(self, mask):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)

    def transform_att(self, att):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError ("dtype must be np.ndarray or torch.Tensor")


class ToTensorAndJitter(TransformBase):
    """Convert to a Tensor and jitter brightness"""
    def __init__(self, brightness_jitter=0.0, normalize=True):
        super().__init__()
        self.brightness_jitter = brightness_jitter
        self.normalize = normalize

    def roll(self):
        return np.random.uniform(max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

    def transform_image(self, image, brightness_factor):
        # handle numpy array
        image = torch.from_numpy(image.transpose((2, 0, 1)))

        # backward compatibility
        if self.normalize:
            return image.float().mul(brightness_factor/255.0).clamp(0.0, 1.0)
        else:
            return image.float().mul(brightness_factor).clamp(0.0, 255.0)

    def transform_mask(self, mask, brightness_factor):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)
        else:
            return mask
    def transform_att(self, att, brightness_factor):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError ("dtype must be np.ndarray or torch.Tensor")


class Normalize(TransformBase):
    """Normalize image"""
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def transform_image(self, image):
        return tvisf.normalize(image, self.mean, self.std, self.inplace)


class ToGrayscale(TransformBase):
    """Converts image to grayscale with probability"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability
        self.color_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_grayscale):
        if do_grayscale:
            if torch.is_tensor(image):
                raise NotImplementedError('Implement torch variant.')
            img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            return np.stack([img_gray, img_gray, img_gray], axis=2)
            # return np.repeat(np.sum(img * self.color_weights, axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
        return image


class ToBGR(TransformBase):
    """Converts image to BGR"""
    def transform_image(self, image):
        if torch.is_tensor(image):
            raise NotImplementedError('Implement torch variant.')
        img_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return img_bgr


class RandomHorizontalFlip(TransformBase):
    """Horizontally flip image randomly with a probability p."""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_flip):
        if do_flip:
            if torch.is_tensor(image):
                return image.flip((2,))
            return np.fliplr(image).copy()
        return image

    def transform_coords(self, coords, image_shape, do_flip):
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1,:] = (image_shape[1] - 1) - coords[1,:]
            return coords_flip
        return coords

    def transform_mask(self, mask, do_flip):
        if do_flip:
            if torch.is_tensor(mask):
                return mask.flip((-1,))
            return np.fliplr(mask).copy()
        return mask

    def transform_att(self, att, do_flip):
        if do_flip:
            if torch.is_tensor(att):
                return att.flip((-1,))
            return np.fliplr(att).copy()
        return att


class RandomHorizontalFlip_Norm(RandomHorizontalFlip):
    """Horizontally flip image randomly with a probability p.
    The difference is that the coord is normalized to [0,1]"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def transform_coords(self, coords, image_shape, do_flip):
        """we should use 1 rather than image_shape"""
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1,:] = 1 - coords[1,:]
            return coords_flip
        return coords
