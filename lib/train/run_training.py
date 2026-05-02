# 系统与标准库导入
import os
import sys
import argparse
import importlib
import random

# 第三方库导入
import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed as dist

# 项目路径与模块导入
import _init_paths
import lib.train.admin.settings as ws_settings

# 全局 CUDNN 设置 (默认关闭 benchmark 以保证初始状态受控)
torch.backends.cudnn.benchmark = False

TRAIN_SCRIPT_REGISTRY = {
    "ostrack": "lib.train.train_script",
    "ugtrack": "lib.train.train_script_ugtrack",
}


# 加载配置参数，设置文件路径，启动训练脚本
def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1, save_dir=None, base_seed=None,
                 use_lmdb=False, script_name_prv=None, config_name_prv=None, use_wandb=False,
                 distill=None, script_teacher=None, config_teacher=None):
    """运行训练脚本。
    参数说明:
        script_name: "experiments/" 文件夹中的实验名称。
        config_name: "experiments/<script_name>" 文件夹中的 yaml 配置文件名。
        cudnn_benchmark: 是否使用 cudnn benchmark (默认为 True)。
        local_rank: 分布式训练中的节点 rank，单卡训练通常为 -1。
        save_dir: 保存 checkpoint 和日志的目录。
        use_lmdb: 是否使用 LMDB 格式的数据集。
        use_wandb: 是否启用 wandb 实验记录。
    """
    
    # ====================
    # 相关设置
    # ====================
    # 如果没有提供 save_dir，则使用默认目录。
    if save_dir is None:
        print("save_dir dir is not given. Use the default dir instead.")
    # 为了避免与 opencv 相关的奇怪崩溃
    cv.setNumThreads(0)
    # 设置 cudnn benchmark 模式
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # ====================
    # 打印实验信息
    # ====================
    print('script_name: {}.py  config_name: {}.yaml'.format(script_name, config_name))

    # ====================
    # 环境随机种子设置
    # ====================
    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)

    # ====================
    # 初始化实验设置 (Settings)
    # ====================
    # 创建训练设置对象
    settings = ws_settings.Settings()
    # 设置实验名称
    settings.script_name = script_name
    # 设置配置文件名称
    settings.config_name = config_name
    # 获取项目根目录的绝对路径
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # 设置实验配置文件的完整路径
    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, config_name))
    # 设置保存目录的绝对路径
    settings.save_dir = os.path.abspath(save_dir)
    # 设置当前进程的本地 GPU 编号
    settings.local_rank = local_rank
    # 是否使用 LMDB 格式的数据集
    settings.use_lmdb = use_lmdb
    # 是否启用 wandb 实验记录
    settings.use_wandb = use_wandb
    # 设置项目存储路径（用于日志和模型保存）
    settings.project_path = 'train/{}/{}'.format(script_name, config_name)
    if script_name_prv is not None and config_name_prv is not None:
        settings.project_path_prv = 'train/{}/{}'.format(script_name_prv, config_name_prv)
    
    # ====================
    # 蒸馏与训练脚本加载
    # ====================
    if distill:
        # 配置蒸馏相关的教师模型参数
        settings.distill = distill
        settings.script_teacher = script_teacher
        settings.config_teacher = config_teacher
        if script_teacher is not None and config_teacher is not None:
            settings.project_path_teacher = 'train/{}/{}'.format(script_teacher, config_teacher)
        settings.cfg_file_teacher = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_teacher, config_teacher))
        # 动态导入蒸馏训练模块
        expr_module = importlib.import_module('lib.train.train_script_distill')
    else:
        # 动态导入常规训练模块。运行前需要先在终端执行: conda activate ostrack
        # ostrack 或者 ugtrack
        expr_module_name = TRAIN_SCRIPT_REGISTRY.get(script_name)
        if expr_module_name is None:
            raise ValueError("Unsupported script name: {}".format(script_name))
        expr_module = importlib.import_module(expr_module_name)

    # ====================
    # 执行训练
    # ====================
    # 获取并运行模块中的 run 函数
    expr_func = getattr(expr_module, 'run')
    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, required=True, help='Name of the train script.')
    parser.add_argument('--config', type=str, required=True, help="Name of the config file.")
    parser.add_argument('--save_dir', type=str, help='the directory to save checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42, help='seed for random numbers')
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb    
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--script_prv', type=str, default=None, help='Name of the train script of previous model.')
    parser.add_argument('--config_prv', type=str, default=None, help="Name of the config file of previous model.")

    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')

    args = parser.parse_args()

    # ====================
    # GPU与分布式环境初始化
    # ====================
    if args.local_rank != -1:
        # 初始化分布式训练进程组 (NCCL后端)
        dist.init_process_group(backend='nccl')
        # 为当前进程绑定指定的本地 GPU
        torch.cuda.set_device(args.local_rank)
    else:
        # 单卡模式：默认使用 0 号卡
        torch.cuda.set_device(0)

    # ====================
    # 启动训练流程
    # ====================
    run_training(args.script, args.config, cudnn_benchmark=args.cudnn_benchmark,
                 local_rank=args.local_rank, save_dir=args.save_dir, base_seed=args.seed,
                 use_lmdb=args.use_lmdb, script_name_prv=args.script_prv, config_name_prv=args.config_prv,
                 use_wandb=args.use_wandb,
                 distill=args.distill, script_teacher=args.script_teacher, config_teacher=args.config_teacher)


if __name__ == '__main__':
    main()
