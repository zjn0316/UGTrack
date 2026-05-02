"""CE backbone with UWB token compatibility.

Extends VisionTransformerCE to support [Template, Search, UWB] token injection.
UWB token is protected from candidate elimination by pruning only the search
span and keeping the tail UWB token in the sequence.

Intended use (per AGENTS.md rule #8):
  This replaces lib/models/ostrack/vit_ce.py for UGTrack Stage-2,
  without modifying the original OSTrack files.
"""

import math

import torch
from torch import nn
from timm.models.layers import trunc_normal_

from lib.models.ostrack.vit_ce import VisionTransformerCE
from lib.models.ostrack.utils import combine_tokens, recover_tokens


class VisionTransformerCEUWB(VisionTransformerCE):
    """CE backbone compatible with UWB token injection.

    UWB token is appended after [Template, Search]. CE candidate elimination
    keeps the original template/search semantics and excludes the tail UWB
    token from search pruning.

    Behaves identically to VisionTransformerCE when uwb_token is None.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uwb_pruner = None
        self.uwb_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.uwb_pos_embed, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {"uwb_pos_embed"}

    @staticmethod
    def _candidate_elimination_keep_tail(attn, tokens, lens_t, lens_s,
                                         keep_ratio, global_index,
                                         box_mask_z):
        """Prune only search tokens while preserving tail tokens.

        仅剪枝 search token，保留末尾的 UWB token。
        """
        lens_keep = math.ceil(keep_ratio * lens_s)
        if lens_keep == lens_s:
            return tokens, global_index, None

        bs, hn, _, _ = attn.shape
        attn_t = attn[:, :, :lens_t, lens_t:lens_t + lens_s]

        if box_mask_z is not None:
            box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(
                -1, attn_t.shape[1], -1, attn_t.shape[-1])
            attn_t = attn_t[box_mask_z]
            attn_t = attn_t.view(bs, hn, -1, lens_s)
            attn_t = attn_t.mean(dim=2).mean(dim=1)
        else:
            attn_t = attn_t.mean(dim=2).mean(dim=1)

        sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)
        topk_idx = indices[:, :lens_keep]
        non_topk_idx = indices[:, lens_keep:]

        keep_index = global_index.gather(dim=1, index=topk_idx)
        removed_index = global_index.gather(dim=1, index=non_topk_idx)

        tokens_t = tokens[:, :lens_t]
        tokens_s = tokens[:, lens_t:lens_t + lens_s]
        tokens_tail = tokens[:, lens_t + lens_s:]

        B, _, C = tokens_s.shape
        attentive_tokens = tokens_s.gather(
            dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
        tokens_new = torch.cat([tokens_t, attentive_tokens, tokens_tail], dim=1)

        return tokens_new, keep_index, removed_index

    def _forward_ce_block_keep_tail(self, blk, x, global_index_t, global_index_s,
                                    mask=None, ce_template_mask=None,
                                    keep_ratio_search=None):
        x_attn, attn = blk.attn(blk.norm1(x), mask, True)
        x = x + blk.drop_path(x_attn)
        lens_t = global_index_t.shape[1]
        lens_s = global_index_s.shape[1]

        removed_index_search = None
        if blk.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = blk.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_s, removed_index_search = self._candidate_elimination_keep_tail(
                attn, x, lens_t, lens_s, keep_ratio_search,
                global_index_s, ce_template_mask)

        x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
        return x, global_index_t, global_index_s, removed_index_search, attn

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,
                         uwb_token=None,
                         pred_uv=None,
                         uwb_conf_pred=None):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        # attention mask handling
        if mask_z is not None and mask_x is not None:
            mask_z = torch.nn.functional.interpolate(
                mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = torch.nn.functional.interpolate(
                mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            if uwb_token is not None:
                uwb_mask = torch.zeros(
                    mask_x.shape[0], 1, 1,
                    device=mask_x.device,
                    dtype=mask_x.dtype)
                mask_x = torch.cat([mask_x, uwb_mask], dim=1)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        global_index_s = torch.arange(lens_x, device=x.device).unsqueeze(0).repeat(B, 1)
        uwb_layer0_removed_index_s = None
        uwb_keep_ratio = 1.0
        # UWB 引导的搜索 token 剪枝（Layer‑0 级）
        if self.uwb_pruner is not None and pred_uv is not None:
            x, global_index_s, uwb_layer0_removed_index_s, uwb_keep_ratio = self.uwb_pruner(
                x, pred_uv, uwb_conf_pred
            )

        # 拼接[模板、搜索、UWB]
        has_uwb = uwb_token is not None
        if has_uwb:
            if uwb_token.ndim == 2:
                uwb_token = uwb_token.unsqueeze(1)
            # pos_embedding
            uwb_token = uwb_token + self.uwb_pos_embed.to(device=uwb_token.device, dtype=uwb_token.dtype)
            if self.cat_mode != "direct":
                raise NotImplementedError("VisionTransformerCEUWB supports UWB token with direct cat_mode only")
            x = torch.cat([z, x, uwb_token], dim=1)
        else:
            x = combine_tokens(z, x, mode=self.cat_mode)

        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        # CE index tracking
        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        removed_indexes_s = [uwb_layer0_removed_index_s] if uwb_layer0_removed_index_s is not None else []
        for i, blk in enumerate(self.blocks):
            if has_uwb:
                x, global_index_t, global_index_s, removed_index_s, attn = \
                    self._forward_ce_block_keep_tail(
                        blk, x, global_index_t, global_index_s, mask_x,
                        ce_template_mask, ce_keep_rate)
            else:
                x, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(x, global_index_t, global_index_s, mask_x,
                        ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc and removed_index_s is not None:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        # ---- UWB: split z, x, uwb ----
        if has_uwb:
            x_all = x
            z = x_all[:, :lens_z]                         # template tokens
            x = x_all[:, lens_z:lens_z + lens_x_new]       # search tokens
            uwb_out = x_all[:, lens_z + lens_x_new:]       # UWB token(s)
        else:
            z = x[:, :lens_z_new]
            x = x[:, lens_z_new:]

        # ---- recover pruned search tokens ----
        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros(
                [B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(
                dim=1,
                index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                src=x)

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        # ---- UWB: discard before OSTrack head ----
        # UWB participates in ViT fusion, but the head expects the last 256
        # tokens to be the restored search grid.
        # UWB 参与 ViT 内部融合；输出给 OSTrack head 前丢弃，保证最后 256 个 token 是 search。
        if has_uwb:
            x = torch.cat([z, x], dim=1)
        else:
            x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
            "uwb_layer0_removed_indexes_s": uwb_layer0_removed_index_s,
            "uwb_prune_keep_ratio": uwb_keep_ratio,
            "uwb_prune_keep_tokens": global_index_s.shape[1],
        }

        return x, aux_dict

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
            uwb_token=uwb_token,
            pred_uv=pred_uv,
            uwb_conf_pred=uwb_conf_pred)
        return x, aux_dict


def _create_vision_transformer_ce_uwb(pretrained=False, **kwargs):
    model = VisionTransformerCEUWB(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_ce_uwb(pretrained=False, **kwargs):
    """ViT-B/16 CE backbone with UWB token compatibility."""
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return _create_vision_transformer_ce_uwb(
        pretrained=pretrained, **model_kwargs)


def vit_large_patch16_224_ce_uwb(pretrained=False, **kwargs):
    """ViT-L/16 CE backbone with UWB token compatibility."""
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    return _create_vision_transformer_ce_uwb(
        pretrained=pretrained, **model_kwargs)
