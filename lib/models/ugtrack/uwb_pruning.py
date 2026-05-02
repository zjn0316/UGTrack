import math

import torch
from torch import nn


class UWBGuidedPruner(nn.Module):
    """Layer-0 UWB-guided search token pruning."""

    def __init__(self,
                 search_size=256,
                 patch_size=16,
                 min_keep_ratio=0.25,
                 max_keep_ratio=1.0,
                 fixed_keep_ratio=0.5,
                 use_conf_dynamic=True):
        super().__init__()
        self.search_size = int(search_size)
        self.patch_size = int(patch_size)
        self.min_keep_ratio = float(min_keep_ratio)
        self.max_keep_ratio = float(max_keep_ratio)
        self.fixed_keep_ratio = float(fixed_keep_ratio)
        self.use_conf_dynamic = bool(use_conf_dynamic)

        grid_size = self.search_size // self.patch_size
        centers = self._build_patch_centers(grid_size)
        self.register_buffer("patch_centers", centers, persistent=False)

    @staticmethod
    def _build_patch_centers(grid_size):
        """构建每个patch的中心坐标（归一化）。
        返回形状为 (N, 2) 的张量，其中 N = grid_size^2，每一行为 (x_center, y_center) 在 [0,1] 范围内。
        """
        coord = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / float(grid_size)
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def _get_keep_ratio(self, uwb_conf_pred):
        """根据UWB置信度计算需要保留的token比例。
        置信度越高，认为预测越可靠，可以保留更少的token（靠近目标即可）；
        置信度越低，保留较多的token（更保守的策略）。
        若uwb_conf_pred为None或未启用动态调整，则返回固定保留比例。
        """
        if uwb_conf_pred is None or not self.use_conf_dynamic:
            return self.fixed_keep_ratio
        conf = uwb_conf_pred.detach().float().reshape(-1).mean().clamp(0.0, 1.0).item()
        # Higher confidence keeps fewer tokens; lower confidence falls back to conservative keep.
        # 置信度越高，保留 token 越少；置信度越低，保留策略越保守。
        keep_ratio = self.max_keep_ratio - conf * (self.max_keep_ratio - self.min_keep_ratio)
        return keep_ratio

    def forward(self, search_tokens, pred_uv, uwb_conf_pred=None):
        """
        前向传播：根据预测的UV坐标（目标中心）筛选保留的token。

        参数:
            search_tokens: 形状 (B, N, C)，搜索区域的token特征，N为patch数量
            pred_uv: 预测的目标中心坐标，形状 (B, 2) 或 (B, ...)，取前两维作为UV坐标，值域[0,1]
            uwb_conf_pred: 可选，UWB预测的置信度，形状任意，用于动态调整保留比例

        返回:
            search_tokens_keep: 保留的token特征，形状 (B, keep_tokens, C)
            keep_index: 保留的token索引，形状 (B, keep_tokens)
            removed_index: 被剔除的token索引，形状 (B, N - keep_tokens)
            keep_ratio: 实际保留比例 (keep_tokens / N)
        """
        if pred_uv is None:
            return search_tokens, None, None, 1.0

        B, N, C = search_tokens.shape
        if N != self.patch_centers.shape[0]:
            raise ValueError(
                "UWBGuidedPruner expected {} search tokens, got {}".format(
                    self.patch_centers.shape[0], N
                )
            )

        pred_uv = pred_uv.detach().float().reshape(B, -1)[:, :2].clamp(0.0, 1.0)
        keep_ratio = max(self.min_keep_ratio, min(self.max_keep_ratio, self._get_keep_ratio(uwb_conf_pred)))
        keep_tokens = int(math.ceil(float(N) * keep_ratio))
        keep_tokens = max(1, min(N, keep_tokens))

        if keep_tokens == N:
            global_index_s = torch.arange(N, device=search_tokens.device).unsqueeze(0).repeat(B, 1)
            return search_tokens, global_index_s, None, 1.0

        centers = self.patch_centers.to(device=search_tokens.device, dtype=pred_uv.dtype)
        dist = torch.cdist(pred_uv.unsqueeze(1), centers.unsqueeze(0).expand(B, -1, -1)).squeeze(1)
        _, keep_index = torch.topk(-dist, k=keep_tokens, dim=1, largest=True, sorted=False)
        keep_index = torch.sort(keep_index, dim=1).values

        all_index = torch.arange(N, device=search_tokens.device).unsqueeze(0).expand(B, -1)
        keep_mask = torch.zeros(B, N, dtype=torch.bool, device=search_tokens.device)
        keep_mask.scatter_(1, keep_index, True)
        removed_index = all_index[~keep_mask].view(B, N - keep_tokens)

        search_tokens_keep = search_tokens.gather(
            dim=1,
            index=keep_index.unsqueeze(-1).expand(B, -1, C),
        )
        return search_tokens_keep, keep_index, removed_index, keep_tokens / float(N)


def build_uwb_guided_pruner(cfg):
    """根据配置字典构建UWBGuidedPruner实例。"""
    backbone_cfg = cfg.MODEL.BACKBONE
    return UWBGuidedPruner(
        search_size=int(cfg.DATA.SEARCH.SIZE),
        patch_size=int(backbone_cfg.STRIDE),
        min_keep_ratio=float(getattr(backbone_cfg, "UWB_PRUNE_MIN_KEEP_RATIO", 0.25)),
        max_keep_ratio=float(getattr(backbone_cfg, "UWB_PRUNE_MAX_KEEP_RATIO", 1.0)),
        fixed_keep_ratio=float(getattr(backbone_cfg, "UWB_PRUNE_KEEP_RATIO", 0.5)),
        use_conf_dynamic=bool(getattr(backbone_cfg, "UWB_PRUNE_CONF_DYNAMIC", True)),
    )
