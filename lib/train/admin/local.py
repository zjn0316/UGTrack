import os


class EnvironmentSettings:
    def __init__(self):
        project_root = '/home/zjn/OSTrack'

        # EN: Use paths relative to the repository root for Linux portability.
        # 中文：使用仓库根目录相对路径，避免 Linux 环境依赖 Windows 盘符。
        self.workspace_dir = project_root    # Base directory for saving network checkpoints.
        self.tensorboard_dir = os.path.join(project_root, "tensorboard")    # Directory for tensorboard files.
        self.pretrained_networks = os.path.join(project_root, "pretrained_models")
        self.otb100_uwb_dir = os.path.join(project_root, "data", "OTB100_UWB")
        self.custom_dataset_dir = os.path.join(project_root, "data", "CustomDataset")
        self.uav123_uwb_dir = os.path.join(project_root, "data", "UAV123_UWB")
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.got10k_val_dir = ''
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = ''
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
