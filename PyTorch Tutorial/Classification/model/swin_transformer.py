"""
构建Swin transformer
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96,norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _,_,H,W = x.shape

        # 高宽必须是patch_size的整数倍，若不是，需要padding
        if (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0 ):
            x = F.pad(x, (0,self.patch_size[1] - W % self.patch_size[1],
                          0,self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        _,_H,W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(1).transpose(1,2)
        x = self.norm(x)

        return x