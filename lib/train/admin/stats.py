


# 统计数值的简单类，用于记录和管理数值的历史
class StatValue:
    def __init__(self):
        self.clear()  # 初始化时清空所有数据

    def reset(self):
        """
        重置当前值为0。
        """
        self.val = 0

    def clear(self):
        """
        清空历史记录并重置当前值。
        """
        self.reset()
        self.history = []

    def update(self, val):
        """
        更新当前值，并将其添加到历史记录中。
        :param val: 新的数值
        """
        self.val = val
        self.history.append(self.val)



# 平均数统计器，记录当前值、平均值、总和和计数
class AverageMeter(object):
    """
    计算并存储当前值、平均值、总和和计数。
    常用于训练过程中统计损失、准确率等指标。
    """
    def __init__(self):
        self.clear()  # 初始化时清空所有数据
        self.has_new_data = False  # 标记是否有新数据

    def reset(self):
        """
        重置所有统计量。
        """
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def clear(self):
        """
        清空历史记录并重置所有统计量。
        """
        self.reset()
        self.history = []

    def update(self, val, n=1):
        """
        更新当前值，并累加总和与计数，重新计算平均值。
        :param val: 新的数值
        :param n: 该数值的权重（出现次数）
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def new_epoch(self):
        """
        新的epoch开始时，将当前平均值保存到历史记录，并重置统计量。
        """
        if self.count > 0:
            self.history.append(self.avg)
            self.reset()
            self.has_new_data = True
        else:
            self.has_new_data = False



def topk_accuracy(output, target, topk=(1,)):
    """
    计算指定k值下的top-k准确率。
    :param output: 模型输出的分数张量，形状为(batch_size, num_classes)
    :param target: 真实标签，形状为(batch_size,)
    :param topk: 需要计算的top-k列表或单个整数
    :return: 对应top-k的准确率（百分比），如果只传入一个k则返回单个值，否则返回列表
    """
    single_input = not isinstance(topk, (tuple, list))
    if single_input:
        topk = (topk,)

    maxk = max(topk)
    batch_size = target.size(0)

    # 取每个样本top-k的预测类别
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # 判断top-k中是否包含正确标签
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # 统计前k个预测中正确的数量
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)[0]
        res.append(correct_k * 100.0 / batch_size)

    if single_input:
        return res[0]

    return res
