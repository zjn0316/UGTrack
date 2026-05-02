import os
import argparse
import random
# 已吃透
# 解析命令行参数，执行python lib/train/run_training.py
def parse_args():
    """
    运行训练脚本。
    参数说明:
        --script: 实验名，对应 "experiments/" 文件夹下的子目录名。
        --config: 配置文件名，对应 "experiments/<script_name>" 下的 yaml 文件（不含扩展名）。
        --save_dir: 保存 checkpoint、日志和 tensorboard 的根目录。
        --mode: 训练模式，可选 "single" (单卡), "multiple" (单机多卡), "multi_node" (多机)。
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    # 实验的跟踪器名
    parser.add_argument('--script', type=str, help='training script name')
    # 配置文件名
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    # 保存结果的根目录
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    # 训练模式
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="multiple", help="train on single gpu or multiple gpus")
    # 每台机器使用的 GPU 数量
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    # 是否使用 LMDB 数据格式
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    # 上一阶段/前一个模型”的脚本名
    parser.add_argument('--script_prv', type=str, help='training script name')
    # 上一阶段/前一个模型”的配置文件名
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')
    # 是否启用 wandb 记录
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb
    
    # for knowledge distillation
    # 是否启用知识蒸馏
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    # 教师模型脚本名
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    # 教师模型配置名
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')

    # for multiple machines
    # 当前机器/进程的节点 rank（多机训练用）。仅 multi_node 需要。
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    # 总进程/节点规模（你这份代码语义上用于 nnodes）。仅 multi_node 需要。
    parser.add_argument('--world-size', type=int, help='Number of processes participating in the job.')
    # 主节点IP。仅 multi_node 需要。
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    # 主节点端口
    parser.add_argument('--port', type=int, default='20000', help='Port of the current rank 0.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.mode == "single":
        train_cmd = "python lib/train/run_training.py --script %s --config %s --save_dir %s --use_lmdb %d " \
                    "--script_prv %s --config_prv %s --distill %d --script_teacher %s --config_teacher %s --use_wandb %d"\
                    % (args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv,
                       args.distill, args.script_teacher, args.config_teacher, args.use_wandb)
    elif args.mode == "multiple":
        train_cmd = "python -m torch.distributed.launch --nproc_per_node %d --master_port %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s --use_wandb %d " \
                    "--distill %d --script_teacher %s --config_teacher %s" \
                    % (args.nproc_per_node, random.randint(10000, 50000), args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
                       args.distill, args.script_teacher, args.config_teacher)
    elif args.mode == "multi_node":
        train_cmd = "python -m torch.distributed.launch --nproc_per_node %d --master_addr %s --master_port %d --nnodes %d --node_rank %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s --use_wandb %d " \
                    "--distill %d --script_teacher %s --config_teacher %s" \
                    % (args.nproc_per_node, args.ip, args.port, args.world_size, args.rank, args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
                       args.distill, args.script_teacher, args.config_teacher)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    os.system(train_cmd)


if __name__ == "__main__":
    main()
