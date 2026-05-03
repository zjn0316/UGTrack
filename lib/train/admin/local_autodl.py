import os


class EnvironmentSettings:
    def __init__(self):
        _root = os.environ.get(
            "UGTRACK_PROJECT_ROOT",
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            ),
        )
        _data = os.environ.get("UGTRACK_DATA_DIR", os.path.join(_root, "data"))

        self.workspace_dir = _root
        self.tensorboard_dir = os.path.join(_root, "tensorboard")
        self.pretrained_networks = os.path.join(_root, "pretrained_models")
        self.otb100_uwb_dir = os.path.join(_data, "OTB100_UWB")
        self.custom_dataset_dir = os.path.join(_data, "CustomDataset")
        self.uav123_uwb_dir = os.path.join(_data, "UAV123_UWB")
        self.lasot_dir = ""
        self.got10k_dir = ""
        self.got10k_val_dir = ""
        self.lasot_lmdb_dir = ""
        self.got10k_lmdb_dir = ""
        self.trackingnet_dir = ""
        self.trackingnet_lmdb_dir = ""
        self.coco_dir = ""
        self.coco_lmdb_dir = ""
        self.lvis_dir = ""
        self.sbd_dir = ""
        self.imagenet_dir = ""
        self.imagenet_lmdb_dir = ""
        self.imagenetdet_dir = ""
        self.ecssd_dir = ""
        self.hkuis_dir = ""
        self.msra10k_dir = ""
        self.davis_dir = ""
        self.youtubevos_dir = ""
