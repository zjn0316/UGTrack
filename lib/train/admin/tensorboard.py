import os
from collections import OrderedDict
# 尝试导入 PyTorch 原生的 Tensorboard 支持，否则回退到 tensorboardX
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print('警告：您的 PyTorch 版本过旧，正在使用 tensorboardX。')
    from tensorboardX import SummaryWriter


class TensorboardWriter:
    """Tensorboard 日志记录类，用于在训练过程中记录多项指标。"""

    def __init__(self, directory, loader_names):
        """
        参数:
            directory - 日志文件保存的根目录
            loader_names - 数据加载器名称列表（如 ['train', 'val']），为每个 loader 创建独立的写入器
        """
        self.directory = directory
        # 为每个 loader (train/val) 创建一个专属的子目录和类对象，方便在界面中对比
        self.writer = OrderedDict({name: SummaryWriter(os.path.join(self.directory, name)) for name in loader_names})

    def write_info(self, script_name, description):
        """记录实验的基础元信息（脚本名和描述）。"""
        tb_info_writer = SummaryWriter(os.path.join(self.directory, 'info'))
        tb_info_writer.add_text('Script_name', script_name)
        tb_info_writer.add_text('Description', description)
        tb_info_writer.close()

    def write_epoch(self, stats: OrderedDict, epoch: int, ind=-1):
        """在每个 Epoch 结束后记录统计的各项指标。
        参数:
            stats - 包含各个 loader 统计数据的有序字典
            epoch - 当前轮次
            ind - 取历史记录中的索引位置位置，默认为最后一项 (-1)
        """
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue
            for var_name, val in loader_stats.items():
                # 如果统计项包含数据历史记录，且标明有新数据，则将其写入 Tensorboard
                if hasattr(val, 'history') and getattr(val, 'has_new_data', True):
                    self.writer[loader_name].add_scalar(var_name, val.history[ind], epoch)