import os

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    project_root = '/home/zjn/OSTrack'
    output_root = os.path.join(project_root, "output")

    # EN: Use repository-relative paths so tests run on Linux without Windows drive letters.
    # 中文：使用仓库相对路径，避免 Linux 测试依赖 Windows 盘符。
    settings.prj_dir = project_root
    settings.save_dir = output_root
    settings.result_plot_path = os.path.join(output_root, "test", "result_plots")
    settings.results_path = os.path.join(output_root, "test", "tracking_results")    # Where to store tracking results
    settings.segmentation_path = os.path.join(output_root, "test", "segmentation_results")
    settings.network_path = os.path.join(output_root, "test", "networks")    # Where tracking networks are stored.
    settings.otb100_uwb_path = os.path.join(project_root, "data", "OTB100_UWB")
    settings.custom_dataset_dir = os.path.join(project_root, "data", "CustomDataset")
    settings.uav123_uwb_path = os.path.join(project_root, "data", "UAV123_UWB")
    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = ''
    settings.lasot_extension_subset_path_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

