import torch.nn as nn

from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)      # 224 -> (224, 224)
        patch_size = to_2tuple(patch_size)  # 16 -> (16, 16)
        self.img_size = img_size            # (224, 224)
        self.patch_size = patch_size        # (16, 16)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])# (14, 14)
        self.num_patches = self.grid_size[0] * self.grid_size[1]# 196
        self.flatten = flatten
        # 用卷积层切patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # 对图像用卷积进行patch切分，得到 (B, 768, H/patch_size, W/patch_size)
        # [B, 3, 224, 224]->[B, 768, 14, 14]
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # 对每个patch的embedding进行归一化
        x = self.norm(x)
        return x
