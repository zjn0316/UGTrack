import collections
import importlib
import os

import torch
import torch.utils.data.dataloader
# 兼容 PyTorch 2.0+
try:
    from torch._six import string_classes
except ImportError:
    string_classes = str
from lib.utils import TensorDict, TensorList

if float(torch.__version__[:3]) >= 1.9 or len('.'.join((torch.__version__).split('.')[0:2])) > 3:
    int_classes = int
else:
    try:
        from torch._six import int_classes
    except ImportError:
        int_classes = int


_WORKER_COMPAT_CACHE = {}


class _WorkerProbeDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.zeros(1)


def resolve_num_workers(requested_num_workers):
    """Return a usable num_workers value on the current machine."""
    if requested_num_workers <= 0:
        return 0

    if os.name != 'nt':
        return requested_num_workers

    if os.environ.get('OSTRACK_ENABLE_WINDOWS_WORKERS', '0') != '1':
        print(
            "Windows DataLoader multiprocessing is disabled on this machine. "
            f"Requested num_workers={requested_num_workers} will use num_workers=0. "
            "Set OSTRACK_ENABLE_WINDOWS_WORKERS=1 to bypass this safeguard."
        )
        return 0

    cache_key = int(requested_num_workers)
    if cache_key in _WORKER_COMPAT_CACHE:
        return _WORKER_COMPAT_CACHE[cache_key]

    try:
        probe_loader = torch.utils.data.DataLoader(
            _WorkerProbeDataset(),
            batch_size=1,
            num_workers=requested_num_workers,
        )
        probe_iter = iter(probe_loader)
        next(probe_iter)
        resolved_num_workers = requested_num_workers
    except (PermissionError, OSError, RuntimeError) as exc:
        print(
            "DataLoader workers unavailable on this Windows machine "
            f"(requested num_workers={requested_num_workers}, error={type(exc).__name__}: {exc}). "
            "Falling back to num_workers=0."
        )
        resolved_num_workers = 0

    _WORKER_COMPAT_CACHE[cache_key] = resolved_num_workers
    return resolved_num_workers


def _check_use_shared_memory():
    """
    检查是否应该使用共享内存（shared memory）来存储张量。
    通常在多进程数据加载时使用，以减少内存复制开销。
    """
    if hasattr(torch.utils.data.dataloader, '_use_shared_memory'):
        return getattr(torch.utils.data.dataloader, '_use_shared_memory')
    collate_lib = importlib.import_module('torch.utils.data._utils.collate')
    if hasattr(collate_lib, '_use_shared_memory'):
        return getattr(collate_lib, '_use_shared_memory')
    return torch.utils.data.get_worker_info() is not None


def ltr_collate(batch):
    """
    将 batch 中的每个数据字段整理到一个张量中，外层维度为 batch size。
    默认在第 0 维（batch 维度）并行。
    """

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
        # if batch[0].dim() < 4:
        #     return torch.stack(batch, 0, out=out)
        # return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.Mapping):
        return {key: ltr_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate(samples) for samples in transposed])
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [ltr_collate(samples) for samples in transposed]
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))


def ltr_collate_stack1(batch):
    """Puts each data field into a tensor. The tensors are stacked at dim=1 to form the batch"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    # 处理Tensor
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 1)
        # if batch[0].dim() < 4:
        #     return torch.stack(batch, 0, out=out)
        # return torch.cat(batch, 0, out=out)
    # 处理 numpy 数组
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 1)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    # 处理字典/自定义容器
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.Mapping):
        return {key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]}
    # 处理列表/序列
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate_stack1(samples) for samples in transposed])
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [ltr_collate_stack1(samples) for samples in transposed]
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))


class LTRLoader(torch.utils.data.dataloader.DataLoader):
    """
    数据加载器。结合了数据集（dataset）和采样器（sampler），并提供
    对数据集进行单进程或多进程迭代。

    注意：与 PyTorch 默认的 DataLoader 唯一的区别在于，这里提供了一个额外的 stack_dim 选项，
            用于选择在哪一个维度上将数据堆叠（stack）以形成一个 batch。

    参数：
        dataset (Dataset): 从中加载数据的数据集。
        batch_size (int, 可选): 每个 batch 加载多少个样本（默认值：1）。
        shuffle (bool, 可选): 设置为 ``True`` 表示在每个 epoch 重新打乱数据（默认值：False）。
        sampler (Sampler, 可选): 定义从数据集中抽取样本的策略。如果指定，则 ``shuffle`` 必须为 False。
        batch_sampler (Sampler, 可选): 类似于 sampler，但每次返回一个 batch 的索引。与 batch_size, shuffle,
            sampler, 和 drop_last 互斥。
        num_workers (int, 可选): 使用多少个子进程来加载数据。0 表示在主进程中加载数据。（默认值：0）
        collate_fn (callable, 可选): 将样本列表合并以形成 mini-batch。
        stack_dim (int): 堆叠形成 batch 的维度。（默认值：0）
        pin_memory (bool, 可选): 如果为 ``True``，数据加载器将在返回前将张量复制到 CUDA 固定内存（pinned memory）中。
        drop_last (bool, 可选): 设置为 ``True`` 以丢弃最后一个不完整的 batch（如果数据集大小不能被 batch size 整除）。
            如果是 ``False`` 且数据集大小不能被整除，则最后一个 batch 将会更小。（默认值：False）
        timeout (numeric, 可选): 如果为正数，则为从 worker 手机 batch 的超时值。应始终为非负数。（默认值：0）
        worker_init_fn (callable, 可选): 如果不为 None，这将在设置种子之后、数据加载之前，在每个子进程中被调用，
            输入为 worker id（一个在 ``[0, num_workers - 1]`` 范围内的整数）。（默认值：None）

    .. 注意:: 默认情况下，每个 worker 的 PyTorch 种子将被设置为 ``base_seed + worker_id``，
              其中 ``base_seed`` 是主进程使用随机数生成器生成的长整型。然而，其他库的种子（例如 NumPy）
              可能在初始化 worker 时重复，导致每个 worker 返回相同的随机数。
              你可以使用 ``torch.initial_seed()`` 来访问每个 worker 的 PyTorch 种子，并在加载数据前设置其他种子。

    .. 警告:: 如果使用了 ``spawn`` 启动方法，:attr:`worker_init_fn` 不能是一个不可序列化（unpicklable）的对象，
              例如 lambda 函数。
    """

    __initialized = False

    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        if collate_fn is None:
            if stack_dim == 0:
                collate_fn = ltr_collate
            elif stack_dim == 1:
                collate_fn = ltr_collate_stack1
            else:
                raise ValueError('Stack dim no supported. Must be 0 or 1.')

        super(LTRLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                 num_workers, collate_fn, pin_memory, drop_last,
                 timeout, worker_init_fn)

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim
