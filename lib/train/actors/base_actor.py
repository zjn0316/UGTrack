from lib.utils import TensorDict


class BaseActor:
    """ Actor 基类。
    Actor 类负责将数据输入网络并计算损失。
    """
    def __init__(self, net, objective):
        """
        参数:
            net - 待训练的网络模型
            objective - 损失函数 (或包含多个损失函数的字典)
        """
        self.net = net
        self.objective = objective

    def __call__(self, data: TensorDict):
        """ 在每个训练迭代中被调用。
        应当将输入数据通过网络，计算损失，并返回该输入数据的训练统计信息。
        参数:
            data - 包含所有必要数据块的 TensorDict。

        返回:
            loss    - 输入数据的总损失（用于反向传播）
            stats   - 包含详细损失项的字典
        """
        raise NotImplementedError

    def to(self, device):
        """ 将网络转移至指定设备
        参数:
            device - 使用的设备，'cpu' 或 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ 设置网络是否处于训练模式。
        参数:
            mode (True) - 布尔值，指定是否处于训练模式。
        """
        self.net.train(mode)

    def eval(self):
        """ 将网络设置为评估模式。"""
        self.train(False)