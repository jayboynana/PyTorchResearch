"""
构建Swin transformer
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import numpy as np
from typing import Optional

def drop_path_r(x,drop_prob:float = 0,training:bool = False):
    if drop_prob == 0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_r(x, self.drop_prob, self.training)


def window_partition(x, window_size : int):
    """
    将feature map 按照window_size划分成一个个不重叠的window
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

class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer = nn.GELU,drop = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class WindowAttention(nn.Module):
    """
    W-MSA,带有相对位置编码的窗口多头注意力
    支持W-MSA和SW-MSA

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size # [Mh,Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义相对位置编码参数表
        # 每个注意力头都有自己的相对位置编码表, 每个表大小都是[2*M-1,2*M-1],这里直接将每个表变成一列, 每个头的表就是每一列
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0] - 1) * (2*window_size[1] - 1),num_heads)) # [2*Mh-1 * 2*Mw-1,nH]

        # 获取窗口内各个patch之间的相对位置索引, 后续根据这个索引，去相对位置编码参数表中找到相对位置编码
        # 构建坐标
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing='ij')) # [2,Mh,Mw]
        coords_flatten = torch.flatten(coords,1) # [2,Mh*Mw]

        # [2,Mh*Mw,1] - [2,1,Mh*Mw] -> [2,Mh*Mw,Mh*Mw] 广播机制
        relative_coords = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        relative_coords = relative_coords.permute(1,2,0).contiguous() # [Mh*Mw,Mh*Mw,2]
        # x,y坐标分别进行加和乘, 维度2上的第一列表示的就是x坐标, 第二列表示的就是y坐标
        relative_coords[:,:,0] += self.window_size[0] - 1
        relative_coords[:,:,1] += self.window_size[1] - 1
        relative_coords[:,:,0] *= 2 * self.window_size[0] - 1
        # x,y坐标相加
        relative_position_index = relative_coords.sum(-1) # [Mh*Mw,Mh*Mw]
        # 将相对位置索引加入到缓存
        self.register_buffer("relative_position_index",relative_position_index)

        self.qkv = nn.Linear(dim,dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化相对位置编码表
        nn.init.trunc_normal_(self.relative_position_bias_table,std=0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x,mask:Optional[torch.Tensor] = None):
        """
        W-MSA 和 SW-MSA 前向计算过程 其实就是主要按照自注意力的计算公式一步步计算：
        Attention(Q,K,V) = SoftMax(Q * K的转置 / 根号d + B(相对位置编码)) * V
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        Returns: x

        """

        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_,N,C = x.shape
        # 生成qkv
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_,N,3,self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale # 自注意力中的缩放, 先缩放再跟k做计算或计算后缩放都可以
        attn = (q @ k.transpose(-2,-1))

        # 先将relative_position_index给展平，再去relative_position_bias_table中找到相对位置编码
        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous() # [nH, Mh*Mw, Mh*Mw]
        # unsqueeze(0)给相对位置编码最前面新增一个维度, 这样就可以与attn广播相加了
        attn = attn + relative_position_bias.unsqueeze(0)

        # 如果是SW-MSA，就需要mask
        if mask is not None:
            # mask:[nW,Mh*Mw, Mh*Mw]
            nW = mask.shape[0] # num_windows

            # 先给mask新建一个1维度，再在mask.unsqueeze(1)的基础上新建一个0维度
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn.view(-1,self.num_heads,N,N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # attn : [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        # v : [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1,2).reshape(B_,N,C)
        # 再经过proj线性层融合各个自注意力头
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class SwinTransformerBlock(nn.Module):
    """
    Basic Layer 中的swin transformer block

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= shift_size < window_size, 'shift_size must be between 0 and window_size !'

        self.norm1 = norm_layer(dim)

        # Window Attention, W-MSA, SW-MSA
        self.attn = WindowAttention(dim,(self.window_size,self.window_size),
                                    num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

    def forward(self,x,attn_mask):
        H,W = self.H,self.W
        B,L,C = x.shape
        assert L == H * W, 'input feature has wrong size'

        shortcut = x
        x = self.norm1(x)
        x = x.view(B,H,W,C)

        # 把feature map给pad到window size的整数倍
        # 若不需pad，计算pad_r和pad_b时，自会求得是0
        pad_l = pad_t = 0
        pad_r = (self.window_size - H % self.window_size) % self.window_size
        pad_b = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0,0,pad_l,pad_r,pad_t,pad_b))
        _,Hp,Wp,_ = x.shape

        # cyclic shift
        # 如果是 SW-MSA 则需要先进行cyclic shift, 移动上边部分和左边部分
        # torch.roll函数中负号表示从上向下，从左向右移动
        if self.shift_size > 0:
            shifted_x = torch.roll(x,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        else:
            shifted_x = x
            attn_mask = None

        # window partition
        x_windows = window_partition(shifted_x,self.window_size) # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1,self.window_size * self.window_size,C) # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows,mask= attn_mask) # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1,self.window_size, self.window_size,C) # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows,self.window_size,Hp,Wp) # [B,Hp,Wp,C]

        # reverse cyclic shift
        # 要还原特征图的左边部分和上边部分
        # torch.roll中shifts参数为正，表示从从下向上，从右向左移动
        if self.shift_size > 0:
            x = torch.roll(shifted_x,shifts=(self.shift_size,self.shift_size),dims=(1,2))
        else:
            x = shifted_x

        # 将之前pad的部分移除
        if pad_r > 0 or pad_b > 0:
            x = x[:,:H,:W,:].contiguous()

        x = x.view(B,H*W,C)

        # FFN
        x = self.drop_path(x) + shortcut
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    swin transformer 中的一个stage，将它定义成一个Layer
    这里的Layer是包含了该层的swin transformer block 和 下一层的patch merging

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        # 每个stage的swin transformer block
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2) == 0 else self.shift_size, # 根据stage中block的深度，来判断是使用W-MSA还是SW-MSA
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path,list) else drop_path,
                                 norm_layer=norm_layer
                                 )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim = dim,norm_layer = norm_layer)
        else:
            self.downsample = None

    def create_mask(self,x,H,W):
        """ mask 模板，方便后续来做滑动窗口的自注意力操作(SW-MSA)"""

        # 保证高宽是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        # 创建全为0的mask，[1,Hp,Hw,1]
        img_mask = torch.zeros((1,Hp,Wp,1),device=x.device)

        # 创建索引，由于cyclic shift，来自不同区域的patch之间不能做自注意力操作
        # 对特征图的高宽建立索引，从而划分出patch做自注意力的区域
        h_slices = (slice(0,-self.window_size),
                    slice(-self.window_size,-self.shift_size),
                    slice(-self.shift_size,None))
        w_slices = (slice(0,-self.window_size),
                    slice(-self.window_size,-self.shift_size),
                    slice(-self.shift_size,None))

        # 根据slices，对来自相同区域的patch进行编号
        # cnt就是编号
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:,h,w,:] = cnt #
                cnt += 1

        # 以下构建窗口的mask矩阵
        # [nW, Mh, Mw, 1] nW即batch*窗口数量;这里由于输入的img_mask的batch是1，所以nW其实就是窗口数量
        mask_windows = window_partition(img_mask,self.window_size)
        mask_windows = mask_windows.view(-1,self.window_size*self.window_size)  # [nW, Mh*Mw]，每个窗口拉直

        # [nW, 1 ,Mh*Mw] - [nW, Mh*Mw, 1] -> [nW, Mh*Mw, Mh*Mw]
        # unsqueeze使得在指定维度上新建该维度，
        # 二者相减，会涉及广播机制，
        # 前一个张量在维度1上的大小变成和后一个张量在维度1上一样的大小；后一个张量在维度2上的大小变成和前一个张量在维度2上一样的大小，然后相减，
        # 得出的张量，表示在每个窗口内，从行方向看，当前patch和窗口内其他的patch是否来自同一区域，
        # 如果来自同一区域，则为0，否则不为0，这样方便后续去算自注意力，达到mask掩码的效果
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        # [nW, Mh * Mw, Mh * Mw] 不为0用-100代替，为0则为0
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self,x,H,W):
        attn_mask = self.create_mask(x,H,W) # [nW,Mh*Mw,Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H,W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk,x,attn_mask)
            else:
                x = blk(x,attn_mask)

        if self.downsample is not None:
            x = self.downsample(x,H,W)
            H,W = (H+1) // 2, (W+1) // 2

        return x,H,W

class SwinTransformer(nn.Module):
    """
    Swin Transformer
    将以上组件组合构建出Swin Transformer
    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers-1))
        self.mlp_ratio = mlp_ratio

        # 将图像划分成不重叠的patches
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 深度衰减规则,随着深度加深，drop_path_rate逐渐增大
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer), # dim随着layer层数翻倍
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                # 按照depth来取dpr中的数,比如i_layer=0,sum(depths[:i_layer])=0,sum(depths[:i_layer+1])=2,取出dpr[0:2]
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer,
                # 除了最后一层layer，前面的layer都带有后一层的patch merging
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x):
        # x:[B,L,C]
        x,H,W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x,H,W = layer(x,H,W)

        x = self.norm(x) # [B,L,C]
        x = self.avgpool(x.transpose(1,2)) # [B,C,1]
        x = torch.flatten(x,1)
        x = self.head(x)

        return x

def swin_tiny_patch4_window7_224(num_classes:int = 1000,**kwargs):
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2,2,6,2),
        num_heads=(3,6,12,24),
        num_classes=num_classes,
        **kwargs
    )
    return model
