from lib.test.utils import TrackerParams                         # 导入测试阶段用于保存跟踪器参数的容器
import os                                                        # 导入系统路径处理模块
from lib.test.evaluation.environment import env_settings         # 导入环境配置读取函数
from lib.config.ugtrack.config import (                          # 导入UGTrack配置相关对象
    cfg,                                                         # 导入UGTrack配置对象
    update_config_from_file,                                     # 导入yaml配置更新函数
)                                                                # 结束UGTrack配置对象导入


def parameters(yaml_name: str):                                  # 根据指定yaml名称构建测试参数
    params = TrackerParams()                                     # 创建跟踪器参数对象
    prj_dir = env_settings().prj_dir                             # 读取项目根目录
    save_dir = env_settings().save_dir                           # 读取训练输出和权重保存目录
    # 从yaml文件更新默认配置
    yaml_file = os.path.join(                                    # 拼接UGTrack实验yaml配置文件路径
        prj_dir,                                                 # 使用项目根目录
        'experiments/ugtrack/%s.yaml' % yaml_name,               # 使用UGTrack实验配置相对路径
    )                                                            # 结束yaml配置路径拼接
    update_config_from_file(yaml_file)                           # 使用yaml文件覆盖默认配置
    params.cfg = cfg                                             # 将更新后的配置保存到参数对象中
    # print("test config: ", cfg)                                # 调试时打印测试配置

    # 模板区域和搜索区域
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR            # 设置模板区域缩放因子
    params.template_size = cfg.TEST.TEMPLATE_SIZE                # 设置模板图像尺寸
    params.search_factor = cfg.TEST.SEARCH_FACTOR                # 设置搜索区域缩放因子
    params.search_size = cfg.TEST.SEARCH_SIZE                    # 设置搜索图像尺寸

    # 网络权重文件路径
    params.checkpoint = os.path.join(                            # 拼接UGTrack测试权重路径
        save_dir,                                                # 使用训练输出和权重保存目录
        "checkpoints/train/ugtrack/%s/UGTrack_ep%04d.pth.tar" %  # 使用UGTrack权重相对路径模板
        (yaml_name, cfg.TEST.EPOCH),                             # 填入实验名和测试轮次编号
    )                                                            # 结束网络权重路径拼接

    # 是否保存所有查询框
    params.save_all_boxes = False                                # 关闭保存所有查询框的选项

    return params                                                # 返回完整测试参数
