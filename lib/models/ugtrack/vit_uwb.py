"""Non-CE ViT backbone with UGTrack UWB token and layer-0 pruning support."""

import torch
from torch import nn
from timm.models.layers import trunc_normal_

from lib.models.ostrack.utils import combine_tokens, recover_tokens
from lib.models.ostrack.vit import VisionTransformer


class VisionTransformerUWB(VisionTransformer):
    """UGTrack-specific non-CE ViT.

    This keeps the original OSTrack backbone untouched and adds UWB token
    injection plus optional layer-0 search token pruning here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uwb_pruner = None
        self.uwb_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.uwb_pos_embed, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {"uwb_pos_embed"}

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                        ce_template_mask=None, ce_keep_rate=None,
                        return_last_attn=False,
                        uwb_token=None,
                        pred_uv=None,
                        uwb_conf_pred=None):
        # ------------------------------------------------------------
        # 1. 基础嵌入与位置编码
        # ------------------------------------------------------------
        B = x.shape[0]
        x = self.patch_embed(x)          # 搜索区域分块嵌入   [B, N_x, C]
        z = self.patch_embed(z)          # 模板区域分块嵌入   [B, N_z, C]

        # 可选的类别 token（如 ViT 的 cls_token）
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # 各自加上位置编码（模板和搜索使用不同的位置编码）
        z += self.pos_embed_z
        x += self.pos_embed_x

        # 可选：添加分段嵌入（区分模板与搜索）
        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        # ------------------------------------------------------------
        # 2. UWB 引导的搜索 token 剪枝（Layer‑0 级）
        # ------------------------------------------------------------
        lens_z = self.pos_embed_z.shape[1]   # 模板 token 数量
        lens_x = self.pos_embed_x.shape[1]   # 原始搜索 token 数量
        global_index_s = torch.arange(lens_x, device=x.device).unsqueeze(0).repeat(B, 1)  # 所有搜索 token 的初始索引
        removed_index_s = None
        uwb_keep_ratio = 1.0

        # 0层剪枝
        if self.uwb_pruner is not None and pred_uv is not None:
            x, global_index_s, removed_index_s, uwb_keep_ratio = self.uwb_pruner(
                x, pred_uv, uwb_conf_pred
            )
            # 剪枝后：
            #   x              -> [B, keep_tokens, C]  只保留靠近 UWB 中心的 token
            #   global_index_s -> [B, keep_tokens]      这些 token 在原序列中的索引
            #   removed_index_s-> [B, N_x - keep_tokens]被丢弃的 token 索引
            #   uwb_keep_ratio -> 保留比例

        # ------------------------------------------------------------
        # 3. 合并模板与搜索 token（按指定模式，如 cat / add / concat）
        # ------------------------------------------------------------
        x = combine_tokens(z, x, mode=self.cat_mode)   # 通常是 [z 的 token] + [x 的 token]

        # ------------------------------------------------------------
        # 4. 合并 UWB token 
        # ------------------------------------------------------------
        has_uwb = uwb_token is not None
        if has_uwb:
            if uwb_token.ndim == 2:
                uwb_token = uwb_token.unsqueeze(1)          # 确保形状 [B, 1, C]
            uwb_token = uwb_token + self.uwb_pos_embed.to(device=uwb_token.device, dtype=uwb_token.dtype)
            x = torch.cat([x, uwb_token], dim=1)            # 将 UWB token 拼接到序列末尾

        # 如果启用了 cls_token，也拼接上去
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)   # Dropout

        # ------------------------------------------------------------
        # 5. 通过 Transformer 编码器
        # ------------------------------------------------------------
        for blk in self.blocks:
            x = blk(x)          # 输出形状: [B, N_z + keep_tokens + (1 if uwb) + (1 if cls), C]

        # ------------------------------------------------------------
        # 6. 分离模板与搜索特征，并恢复被剪枝的位置
        # ------------------------------------------------------------
        # 提取模板部分的输出（前 lens_z 个 token）
        z = x[:, :lens_z]
        # 提取搜索部分的输出（紧跟着的 token，长度 = global_index_s.shape[1] 即保留的 token 数）
        x_search = x[:, lens_z:lens_z + global_index_s.shape[1]]

        # 若之前进行了 UWB 剪枝，需要将保留的搜索 token 放回原始位置（被丢弃的位置填零）
        if removed_index_s is not None:
            # 创建填充零张量，补齐被丢弃的 token 数量
            pad_x = torch.zeros([B, lens_x - x_search.shape[1], x_search.shape[2]], device=x_search.device)
            x_search = torch.cat([x_search, pad_x], dim=1)
            # 合并保留索引和丢弃索引，得到所有位置索引（保持原始顺序）
            index_all = torch.cat([global_index_s, removed_index_s], dim=1)
            C = x_search.shape[-1]
            # 使用 scatter 将特征放回原始位置（未保留的位置依然是零）
            x_search = torch.zeros_like(x_search).scatter_(
                dim=1,
                index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                src=x_search,
            )

        # 重新拼接模板和恢复后的搜索特征
        x = torch.cat([z, x_search], dim=1)
        # 可选：恢复合并模式（例如将拼接的模板和搜索 token 分开成独立张量，供后续 head 使用）
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        # ------------------------------------------------------------
        # 7. 构建辅助信息字典（用于监控、可视化或后续损失计算）
        # ------------------------------------------------------------
        aux_dict = {
            "attn": None,                                   # 预留注意力图（若 return_last_attn）
            "removed_indexes_s": [removed_index_s] if removed_index_s is not None else [],
            "uwb_layer0_removed_indexes_s": removed_index_s,   # UWB 剪枝丢弃的索引
            "uwb_prune_keep_ratio": uwb_keep_ratio,            # 实际保留比例
            "uwb_prune_keep_tokens": global_index_s.shape[1],  # 保留的 token 数量
        }
        return self.norm(x), aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False,
                uwb_token=None,
                pred_uv=None,
                uwb_conf_pred=None):
        x, aux_dict = self.forward_features(
            z, x,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
            uwb_token=uwb_token,
            pred_uv=pred_uv,
            uwb_conf_pred=uwb_conf_pred,
        )
        return x, aux_dict


def _create_vision_transformer_uwb(pretrained=False, **kwargs):
    model = VisionTransformerUWB(**kwargs)

    if pretrained:
        if "npz" in pretrained:
            model.load_pretrained(pretrained, prefix="")
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint["model"], strict=False)
            print("Load pretrained model from: " + pretrained)

    return model


def vit_base_patch16_224_uwb(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return _create_vision_transformer_uwb(
        pretrained=pretrained, **model_kwargs)
