"""
构建Swin transformer
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size : int):
    """
    将feature map 按照windo_size划分成一个个不重叠的window
    Args:
        x : (B, H, W, C)
        window_size: window size(M)

    Returns:
        windows: (num_windows*B,window_size,window_size,C)
    """

    B, H, W, C = x.shape
    # view: [B, H//Mh, Mh, W//Mw, Mw, C]
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    # permute : [B, H//Mh, Mh, W//Mw, Mw, C] -> [B,H//Mh,W//Mw,Mh,Mw,C]
    # view : [B,H//Mh,W//Mw,Mh,Mw,C] -> [B*num_windows,Mh,Mw,C]
    x = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return x

def window_reverse(windows,window_size : int, H : int, W : int):
    """
    window_partition的反向操作，将一个个window还原成一个feature_map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x : (B, H, W, C)
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view : [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size,-1)

    # permute : [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view : [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    """
    Patch Embedding

    Args:
        patch_size(int) -> ((int,int)) : patch size for input image
        in_chans(int) : Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

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
        _,_,H,W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)

        return x,H,W

class PatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self,dim, norm_layer = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4*dim)
        self.reduction = nn.Linear(4*dim,2*dim,bias=False)

    def forward(self, x, H, W):
        """
        x:[B,H*W,C]
        """
        B,L,C = x.shape
        assert L == H*W, 'input feature has wrong size !'

        x = x.view(B,H,W,C)

        # padding
        # 如果H,W不是2的整数倍，则需要padding
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x,(0,0,0,W % 2,0,H % 2))

        # [B, H/2, W/2, C]
        # 按间隔为2取出对应位置的元素，此时H，W都减半
        x0 = x[:,0::2,0::2,:]
        x1 = x[:,1::2,0::2,:]
        x2 = x[:,0::2,1::2,:]
        x3 = x[:,1::2,1::2,:]

        # 按照通道方向进行拼接
        # [B, H/2, W/2, 4*C]
        x = torch.cat([x0,x1,x2,x3],-1)
        x = x.view(B,-1,4*C)


        x = self.norm(x)

        # 达到了类似卷积神经网络一样的效果
        # 空间上高宽减半，深度上通道翻倍
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x
