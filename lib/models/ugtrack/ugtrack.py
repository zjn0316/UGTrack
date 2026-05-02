import os

import torch
from torch import nn

from lib.models.layers.head import build_box_head
from lib.models.ugtrack.ostrack_uwb import OSTrackUWB
from lib.models.ugtrack.uwb_pruning import build_uwb_guided_pruner
from lib.models.ugtrack.vit_uwb import vit_base_patch16_224_uwb
from lib.models.ugtrack.vit_ce_uwb import (
    vit_base_patch16_224_ce_uwb,
    vit_large_patch16_224_ce_uwb,
)
from lib.models.ugtrack.uwb_branch import build_uwb_branch


class UGTrack(nn.Module):
    """This is the base class for UGTrack."""

    def __init__(self, uwb_branch, tracker=None):
        super().__init__()
        self.uwb_branch = uwb_branch
        self.tracker = tracker

    def forward(self,
                search_uwb_seq,
                template=None,
                search=None,
                stage=1,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False):
        if stage not in [1, 2]:
            raise NotImplementedError

        uwb_out = self.forward_uwb(search_uwb_seq, stage)
        if stage == 1:
            return uwb_out

        out = self.forward_tracker(
            template=template,
            search=search,
            uwb_token=uwb_out["uwb_token"],
            pred_uv=uwb_out["pred_uv"],
            uwb_conf_pred=uwb_out["uwb_conf_pred"],
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
        )
        out.update(uwb_out)
        return out

    def forward_uwb(self, search_uwb_seq, stage):
        return self.uwb_branch(search_uwb_seq)

    def forward_tracker(self,
                        template,
                        search,
                        uwb_token,
                        pred_uv=None,
                        uwb_conf_pred=None,
                        ce_template_mask=None,
                        ce_keep_rate=None,
                        return_last_attn=False):
        if self.tracker is None:
            raise ValueError("UGTrack stage-2 forward requires tracker")
        if template is None or search is None:
            raise ValueError("UGTrack stage-2 forward requires template and search images")

        return self.tracker(
            template=template,
            search=search,
            uwb_token=uwb_token,
            pred_uv=pred_uv,
            uwb_conf_pred=uwb_conf_pred,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
        )


def build_ugtrack(cfg, training=True):
    # =====================
    # 设置视觉分支初始化路径
    # =====================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    pretrained_path = os.path.join(project_root, "pretrained_models")
    pretrain_file = str(getattr(cfg.MODEL, "PRETRAIN_FILE", "") or "")
    pretrain_is_ostrack_checkpoint = (
        "checkpoints/train/ostrack" in pretrain_file.replace("\\", "/").lower()
        or os.path.basename(pretrain_file).startswith("OSTrack_")
    )
    mae_pretrained = ""
    if pretrain_file and training and not pretrain_is_ostrack_checkpoint:
        # EN: Keep absolute paths unchanged; resolve relative model names under pretrained_models.
        # 中文：绝对路径保持不变；相对模型名从 pretrained_models 下解析。
        mae_pretrained = pretrain_file if os.path.isabs(pretrain_file) else os.path.join(pretrained_path, pretrain_file)

    # =====================
    # 构建 UWB branch
    # =====================
    uwb_branch = build_uwb_branch(cfg)

    # =====================
    # 构建 OSTrack branch
    # =====================
    tracker = None
    if int(getattr(cfg.TRAIN, "STAGE", 1)) == 2:
        if cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224":
            backbone = vit_base_patch16_224_uwb(mae_pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1

        elif cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224_ce":
            backbone = vit_base_patch16_224_ce_uwb(mae_pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                                   ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                                   ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1

        elif cfg.MODEL.BACKBONE.TYPE == "vit_large_patch16_224_ce":
            backbone = vit_large_patch16_224_ce_uwb(mae_pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                                    ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                                    ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1

        else:
            raise NotImplementedError

        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
        if bool(getattr(cfg.MODEL.BACKBONE, "UWB_PRUNE_ENABLE", False)):
            backbone.uwb_pruner = build_uwb_guided_pruner(cfg)
        box_head = build_box_head(cfg, hidden_dim)
        tracker = OSTrackUWB(backbone, box_head, aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE)

        # =====================
        # 加载已训练 OSTrack tracker 权重
        # =====================
        if pretrain_file and pretrain_is_ostrack_checkpoint:
            # EN: Stage-2 YAML may use repository-relative checkpoint paths on Linux.
            # 中文：Stage-2 YAML 在 Linux 下可使用仓库相对 checkpoint 路径。
            checkpoint_path = pretrain_file if os.path.isabs(pretrain_file) else os.path.join(project_root, pretrain_file)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            missing_keys, unexpected_keys = tracker.load_state_dict(checkpoint["net"], strict=False)
            print("Load pretrained OSTrack tracker from: {}".format(checkpoint_path))


    # =====================
    # 构建 UGTrack
    # =====================
    return UGTrack(
        uwb_branch=uwb_branch,
        tracker=tracker,
    )
