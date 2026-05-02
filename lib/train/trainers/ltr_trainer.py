# 系统与标准库导入
import os
import datetime
import time
from collections import OrderedDict

# PyTorch 与深度学习库导入
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# 项目本地模块导入
# 训练基础类
from lib.train.trainers import BaseTrainer
# 日志与可视化工具
from lib.train.data.wandb_logger import WandbWriter
from lib.train.admin import AverageMeter, StatValue, TensorboardWriter
# 工具函数
from lib.utils.misc import get_world_size


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        参数:
            actor - 用于训练网络的 Actor 对象
            loaders - 数据加载器列表，例如 [train_loader, val_loader]
            optimizer - 训练使用的优化器（如 Adam）
            settings - 训练配置对象
            lr_scheduler - 学习率调度器
            use_amp - 是否使用自动混合精度训练
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        # 设置默认配置
        self._set_default_settings()

        # 初始化统计变量字典，为每个 loader 创建条目
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # 初始化日志记录工具 (Tensorboard 与 Wandb)
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
            # 设置 Tensorboard 保存路径
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            # 如果启用 Wandb，初始化 WandbWriter
            if settings.use_wandb:
                world_size = get_world_size()
                # 计算当前已训练的样本总数
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                # 计算日志记录间隔
                interval = (world_size * settings.batchsize)
                self.wandb_writer = WandbWriter(settings.project_path[6:], {}, tensorboard_writer_dir, cur_train_samples, interval)

        # 训练过程控制标志
        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        
        # 如果使用混合精度训练，初始化 GradScaler
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        """设置训练器的默认参数值（如果用户未指定）。"""
        # 所有默认值的字典
        default = {'print_interval': 10,   # 每隔多少个迭代打印一次统计信息
                   'print_stats': None,    # 指定要打印的特定统计项（None 表示全部打印）
                   'description': ''}      # 实验描述信息

        for param, default_value in default.items():
            # 如果 settings 中没有该参数，则使用 getattr 获取 None 并用 setattr 设置默认值
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """执行一个周期的训练或验证。"""

        # 根据 loader 类型（训练或验证）切换 Actor 的状态和梯度计算开关
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        # 初始化计时统计
        self._init_timing()

        for i, data in enumerate(loader, 1):
            # 记录数据读取完成时间
            self.data_read_done_time = time.time()
            
            # 将输入数据转移至 GPU
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            # 将当前 epoch 和 settings 存入 data 字典供 Actor 使用
            data['epoch'] = self.epoch
            data['settings'] = self.settings
            
            # 前向传播 (Forward pass)
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                # 使用自动混合精度进行前向传播
                with autocast():
                    loss, stats = self.actor(data)

            # 反向传播与权重更新 (Backward pass)
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    # 梯度裁剪 (Gradient Clipping)
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    # 使用 GradScaler 进行混合精度反向传播
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # 更新统计信息
            batch_size = self._get_batch_size(data, loader.stack_dim)
            self._update_stats(stats, batch_size, loader)

            # 打印统计日志
            self._print_stats(i, loader, batch_size)

            # 更新 Wandb 状态（按 print_interval 间隔记录）
            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
                if self.settings.local_rank in [-1, 0]:
                    self.wandb_writer.write_log(self.stats, self.epoch)

        # 每个 Epoch 结束后打印耗时摘要
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))

    @staticmethod
    def _get_batch_size(data, stack_dim):
        candidate_keys = [
            'template_images',
            'search_images',
            'search_uwb_seq',
            'search_uwb_gt',
            'search_uwb_conf',
        ]
        for key in candidate_keys:
            value = data.get(key)
            if hasattr(value, 'shape'):
                return value.shape[stack_dim]
        raise KeyError("Unable to infer batch size from data keys: {}".format(list(data.keys())))

    def train_epoch(self):
        """为每个数据加载器（Loader）执行一个 Epoch 的训练/验证。"""
        for loader in self.loaders:
            # 检查当前 Epoch 是否达到了该 Loader 的执行间隔
            if self.epoch % loader.epoch_interval == 0:
                # 如果是分布式训练，为采样器设置当前 Epoch 号，以保证数据打乱的随机性
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                # 执行具体的数据循环过程
                self.cycle_dataset(loader)

        # 每个 Epoch 结束后重置/更新统计指标
        self._stats_new_epoch()
        # 仅在主进程中将本轮统计结果写入 Tensorboard
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        """更新统计指标。"""
        # 如果该 loader 的统计字典尚未初始化，则创建对应的 AverageMeter 字典
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # 记录学习率状态（仅针对训练阶段）
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                # 如果是新的学习率组，初始化为 StatValue
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        # 遍历 Actor 返回的所有 loss 和指标并累加更新
        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        """记录并打印训练统计信息（FPS、耗时、Loss 等）。"""
        self.num_frames += batch_size
        current_time = time.time()
        # 计算当前 batch 的 FPS 和整个 Epoch 的平均 FPS
        batch_time = max(current_time - self.prev_time, 1e-12)
        epoch_time = max(current_time - self.start_time, 1e-12)
        batch_fps = batch_size / batch_time
        average_fps = self.num_frames / epoch_time
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        # 累加各阶段耗时
        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)  # 数据读取耗时
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)  # GPU 传输耗时
        self.avg_forward_time += current_time - self.data_to_gpu_time  # 前向计算耗时

        # 达到打印间隔或到达 epoch 末尾时进行打印
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 打印详细耗时统计
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % (epoch_time / self.num_frames * batch_size)

            # 打印各项 Loss 和指标的平均值
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)

            # 输出到控制台并写入日志文件
            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        """在一个新的 Epoch 开始前记录学习率并重置统计器。"""
        # 记录当前每个参数组的学习率
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    # 兼容某些旧版驱动或特定调度器
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        # 遍历所有 loader 的统计项，触发新 Epoch 的初始化（通常是清除旧的累加值）
        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
