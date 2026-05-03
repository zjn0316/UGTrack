import os

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    _root = os.environ.get(
        "UGTRACK_PROJECT_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    )
    _output = os.environ.get("UGTRACK_OUTPUT_DIR", os.path.join(_root, "output"))
    _data = os.environ.get("UGTRACK_DATA_DIR", os.path.join(_root, "data"))

    settings.prj_dir = _root
    settings.save_dir = _output
    settings.result_plot_path = os.path.join(_output, "test", "result_plots")
    settings.results_path = os.path.join(_output, "test", "tracking_results")
    settings.segmentation_path = os.path.join(_output, "test", "segmentation_results")
    settings.network_path = os.path.join(_output, "test", "networks")
    settings.otb100_uwb_path = os.path.join(_data, "OTB100_UWB")
    settings.custom_dataset_dir = os.path.join(_data, "CustomDataset")
    settings.uav123_uwb_path = os.path.join(_data, "UAV123_UWB")
    settings.davis_dir = ""
    settings.got10k_lmdb_path = ""
    settings.got10k_path = ""
    settings.got_packed_results_path = ""
    settings.got_reports_path = ""
    settings.itb_path = ""
    settings.lasot_extension_subset_path_path = ""
    settings.lasot_lmdb_path = ""
    settings.lasot_path = ""
    settings.nfs_path = ""
    settings.otb_path = ""
    settings.tc128_path = ""
    settings.tn_packed_results_path = ""
    settings.tnl2k_path = ""
    settings.tpl_path = ""
    settings.trackingnet_path = ""
    settings.uav_path = ""
    settings.vot18_path = ""
    settings.vot22_path = ""
    settings.vot_path = ""
    settings.youtubevos_dir = ""

    return settings
