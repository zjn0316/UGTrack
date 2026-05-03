import jpeg4py
import cv2 as cv
from PIL import Image
import numpy as np

davis_palette = np.repeat(np.expand_dims(np.arange(0,256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                         [0, 64, 128], [128, 64, 128]]


def default_image_loader(path):
    """默认图像加载器，从给定路径读取图像。优先尝试 jpeg4py_loader，
    若不可用则回退到 opencv_loader。"""
    if default_image_loader.use_jpeg4py is None:
        # 尝试使用 jpeg4py
        im = jpeg4py_loader(path)
        if im is None:
            default_image_loader.use_jpeg4py = False
            print('改用 opencv_loader。')
        else:
            default_image_loader.use_jpeg4py = True
            return im
    if default_image_loader.use_jpeg4py:
        return jpeg4py_loader(path)
    return opencv_loader(path)

default_image_loader.use_jpeg4py = None


def jpeg4py_loader(path):
    """使用 jpeg4py 读取图像 https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def opencv_loader(path):
    """使用 opencv 的 imread 读取图像，并以 RGB 格式返回"""
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)

        # 转换为 RGB 并返回
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def jpeg4py_loader_w_failsafe(path):
    """使用 jpeg4py 读取图像（带故障回退）https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except:
        try:
            im = cv.imread(path, cv.IMREAD_COLOR)

            # 转换为 RGB 并返回
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        except Exception as e:
            print('ERROR: Could not read image "{}"'.format(path))
            print(e)
            return None


def opencv_seg_loader(path):
    """使用 opencv 的 imread 读取分割标注"""
    try:
        return cv.imread(path)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def imread_indexed(filename):
    """根据给定文件名加载索引图像，用于读取分割标注。"""

    im = Image.open(filename)

    annotation = np.atleast_3d(im)[...,0]
    return annotation


def imwrite_indexed(filename, array, color_palette=None):
    """将索引图像保存为 PNG，用于保存分割标注。"""

    if color_palette is None:
        color_palette = davis_palette

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')