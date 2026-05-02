import torch
import torch.nn as nn

from lib.models.layers.uwb_encoder import UWBGRUEncoder, UWBMLPEncoder, UWBTCNEncoder
from lib.models.layers.uwb_head import UWBHead, UWBTokenHead


class UWBBranch(nn.Module):
    """UWB branch used by UGTrack."""

    def __init__(self, encoder, conf_head, pred_head, token_head=None, pred_mode="residual"):
        super().__init__()
        self.encoder = encoder
        self.conf_head = conf_head
        self.pred_head = pred_head
        self.token_head = token_head if token_head is not None else nn.Identity()
        self.pred_mode = str(pred_mode).lower()

    @staticmethod
    def _set_trainable(module, trainable):
        for param in module.parameters():
            param.requires_grad = trainable

    def configure_trainable(self, stage):
        stage = int(stage)
        if stage == 1:
            self._set_trainable(self.encoder, True)
            self._set_trainable(self.pred_head, True)
            self._set_trainable(self.conf_head, True)
            self._set_trainable(self.token_head, False)
            return
        if stage == 2:
            self._set_trainable(self.encoder, False)
            self._set_trainable(self.pred_head, False)
            self._set_trainable(self.conf_head, False)
            self._set_trainable(self.token_head, True)
            return
        raise ValueError("stage must be 1 or 2")

    def forward(self, uwb_seq):
        uwb_seq = uwb_seq.float()
        uwb_feat = self.encoder(uwb_seq)
        pred_delta_uv = self.pred_head(uwb_feat)

        if self.pred_mode == "direct":
            pred_uv = torch.clamp(pred_delta_uv, 0.0, 1.0)
        elif self.pred_mode == "residual":
            pred_uv = torch.clamp(uwb_seq[:, -1, :] + pred_delta_uv, 0.0, 1.0)
        else:
            raise ValueError("UWB_PRED_MODE must be one of ['direct', 'residual']")

        uwb_conf_logit = self.conf_head(uwb_feat)
        uwb_conf_pred = torch.sigmoid(uwb_conf_logit)
        uwb_token = self.token_head(uwb_feat)

        return {
            "uwb_token": uwb_token,
            "pred_uv": pred_uv,
            "pred_delta_uv": pred_delta_uv,
            "uwb_conf_logit": uwb_conf_logit,
            "uwb_conf_pred": uwb_conf_pred,
            "uwb_pred": pred_uv,
            "uwb_delta": pred_delta_uv,
            "uwb_conf": uwb_conf_logit,
            "uwb_pred_conf": uwb_conf_pred,
            "uwb_alpha": uwb_conf_pred,
        }


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _build_uwb_encoder(cfg):
    backbone_cfg = cfg.MODEL.BACKBONE
    encoder_type = str(backbone_cfg.UWB_ENCODER).lower()
    embed_dim = int(backbone_cfg.UWB_EMBED_DIM)

    if encoder_type == "mlp":
        return UWBMLPEncoder(
            in_dim=int(backbone_cfg.UWB_INPUT_DIM),
            seq_len=int(cfg.DATA.UWB.SEQ_LEN),
            hidden_dims=_as_list(backbone_cfg.UWB_MLP_HIDDEN_DIMS),
            out_dim=embed_dim,
            dropout=float(getattr(backbone_cfg, "UWB_MLP_DROPOUT", 0.1)),
        )

    if encoder_type == "gru":
        return UWBGRUEncoder(
            in_dim=int(backbone_cfg.UWB_INPUT_DIM),
            input_proj_dim=int(getattr(backbone_cfg, "UWB_GRU_INPUT_PROJ_DIM", 64)),
            hidden_dim=int(getattr(backbone_cfg, "UWB_GRU_HIDDEN_DIM", embed_dim)),
            out_dim=embed_dim,
            dropout=float(getattr(backbone_cfg, "UWB_GRU_DROPOUT", 0.1)),
        )

    if encoder_type in ["tcn", "conv1d"]:
        return UWBTCNEncoder(
            in_dim=int(backbone_cfg.UWB_INPUT_DIM),
            channels=int(backbone_cfg.UWB_TCN_CHANNELS),
            dilations=_as_list(backbone_cfg.UWB_TCN_DILATIONS),
            out_dim=embed_dim,
            kernel_size=int(backbone_cfg.UWB_TCN_KERNEL_SIZE),
            dropout=float(backbone_cfg.UWB_TCN_DROPOUT),
        )

    raise NotImplementedError


def build_uwb_branch(cfg):
    head_cfg = cfg.MODEL.HEAD
    embed_dim = int(cfg.MODEL.BACKBONE.UWB_EMBED_DIM)
    uwb_encoder = _build_uwb_encoder(cfg)

    token_head_name = str(head_cfg.UWB_TOKEN_HEAD).lower()
    if token_head_name == "identity":
        uwb_token_head = nn.Identity()
    elif token_head_name == "mlp":
        uwb_token_head = UWBTokenHead(in_dim=embed_dim, token_dim=int(head_cfg.UWB_TOKEN_DIM))
    else:
        raise NotImplementedError

    dropout = float(getattr(head_cfg, "UWB_HEAD_DROPOUT", 0.1))
    branch = UWBBranch(
        encoder=uwb_encoder,
        conf_head=UWBHead(in_dim=embed_dim, task_dim=1, dropout=dropout, final_act=None),
        pred_head=UWBHead(in_dim=embed_dim, task_dim=2, dropout=dropout, final_act=None),
        token_head=uwb_token_head,
        pred_mode=head_cfg.UWB_PRED_MODE,
    )
    branch.configure_trainable(getattr(cfg.TRAIN, "STAGE", 1))
    return branch
