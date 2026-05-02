"""UGTrack-specific OSTrack wrapper.

The original lib.models.ostrack.OSTrack is kept unchanged. This wrapper only
adds the extra UWB prior arguments needed by UGTrack backbones.
"""

import torch

from lib.models.ostrack.ostrack import OSTrack


class OSTrackUWB(OSTrack):
    def forward(self,
                template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                uwb_token=None,
                pred_uv=None,
                uwb_conf_pred=None):
        x, aux_dict = self.backbone(
            z=template,
            x=search,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
            uwb_token=uwb_token,
            pred_uv=pred_uv,
            uwb_conf_pred=uwb_conf_pred,
        )

        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out["backbone_feat"] = x
        return out
