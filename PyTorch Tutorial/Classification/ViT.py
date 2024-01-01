"""
This is ViT that Jayce built again for practice
根据github上的代码再次复现一下
后面用中文，方便阅读
transformer系列很重要的就是理解维度，ViT很重要的就是理解其中维度的变化
理解好维度，整个架构就清晰了
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):

    """
    这是ViT的开头，将图片打成patch，进行所谓的Linear projection，
    这个投影是用卷积操作的，投影后，空间和通道都有变化，空间变成14*14，通道变成768
    """

    def __init__(self,image_size = 224,patch_size = 16,in_channels = 3,embed_dim = 768,norm_layer = None):
        super().__init__()
        image_size = (image_size,image_size)
        patch_size = (patch_size,patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0],image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0]*self.grid_size[1]

        self.proj = nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):

        B,C,H,W = x.shape

        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input size ({H}*{W}) doesn't match ({self.image_size[0]}*{self.image_size[1]})"

        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        x = self.norm_layer(x)

        return x


class Attention(nn.Module):

    """
    做自注意力，其中的维度变化需思考，但做完自注意力后，输出的维度与输入是保持一直的！
    """

    def __init__(self,
                 dim,
                 num_heads = 8,
                 qkv_bias = False,
                 qk_scale = None,
                 attn_drop_ratio = 0.,
                 proj_drop_ratio = 0.,
                 ):
        super(Attention,self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv_scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim,dim*3,bias = qkv_bias) #q，k，v是根据输入，通过线性层生成的，维度是原来维度的三倍，三倍得到的就是q，k，v
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self,x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B,N,C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head] 这里交换维度，使得可以取出q，k，v
        qkv = self.qkv(x)
        qkv = qkv.reshape(B,N,3,self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2,0,3,1,4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head] 也就是qkv的维度
        q,k,v = qkv[0],qkv[1],qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        # 这里张量在做乘运算的时候，其实是最后两个维度的运算
        attention = (q @ k.transpose(-1,-2)) * self.qkv_scale
        attention = attention.softmax(dim = -1)
        attention = self.attn_drop(attention)

        # @: multiply -> [batch_size, num_heads, num_patches + 1,embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head] 交换维度，这样方便将每个头拼接起来
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim] 最后得出的维度与输入的x维度是一致的
        x = attention @ v
        x = x.transpose(1,2)
        x = x.reshape(B,N,C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):

    def __init__(self,in_features,hidden_features = None,out_features = None,act_layer = nn.GELU,drop = 0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """
    Block将多头注意力模块和MLP模块组合起来构成Transformer Block
    """
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio = 0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio = 0.,
                 mlp_ratio = 4, #这里的系数是MLP中隐层元个数与输入层元的倍数，ViT中一般是4倍
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm
                 ):
        super(Block,self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        MLP_hidden_dim = int(dim*mlp_ratio)
        self.MLP = MLP(in_features=dim,
                       hidden_features=MLP_hidden_dim,
                       act_layer=act_layer,
                       drop = drop_ratio)

    def forward(self,x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.MLP(self.norm2(x)))

class VisionTransformer(nn.Module):

    """
    这个将前面的模块放在一起构建出ViT的整体
    其中有一个参数distilled，只要和它有关系的变量，都是与ViT无关的
    """

    def __init__(self,img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 representation_size=None, distilled=False,
                 drop_ratio=0.,attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed,
                 norm_layer=None, act_layer=None):

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(norm_layer,eps = 1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(image_size=img_size,patch_size=patch_size,in_channels=in_c,
                                      embed_dim=embed_dim,norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+self.tokens,embed_dim))
        self.pos_drop = nn.Dropout(drop_ratio)

        dpr = [x.item() for x in torch.linspace(0,drop_path_ratio,depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,drop_ratio=dpr[i],
                  attn_drop_ratio=attn_drop_ratio,drop_path_ratio=drop_path_ratio,mlp_ratio=mlp_ratio,
                  act_layer=act_layer,norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        #这里是表示层，就是在提取出class token之后，可以再在最后的线性层之前加一层线性层，也可以不加
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc',nn.Linear(embed_dim,representation_size)),
                 ('act',nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        #分类头
        self.head = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        #参数初始化，像分类的token和位置编码都需要参数初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights) #对当前对象调用apply方法，这个方法可以给对象应用一个函数，这里就是对模型应用了权重初始化的函数

    def forward_features(self,x):
        """
        前向提取特征，并且取出了batch中每个样本的分类token
        """
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)
        if self.dist_token is None:
            x = torch.cat((cls_token,x),dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x+self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:,0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self,x):
        """
        对前面提取出的分类token，输入到分类头，进行分类
        """
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT 权重初始化
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):

    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=None,
        num_classes=num_classes
    )

    return model





















