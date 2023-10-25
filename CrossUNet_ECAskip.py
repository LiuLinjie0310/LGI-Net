""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
import numpy as np
from Models.ECAAttention import ECAAttention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=16, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.qk_conv = nn.Conv2d(dim, dim // 4, kernel_size=3, stride=1, padding=1)
        # self.q_conv = nn.Conv2d(dim, dim//4, kernel_size=3, stride=1, padding=1)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[0], qkv[1]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)

        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x

class Self_CSWinBlock(nn.Module):

    def __init__(self, dim, num_heads, patches_resolution=None,
                 split_size=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.patches_resolution = patches_resolution
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            LePEAttention(
                dim // 2, resolution=self.patches_resolution, idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        B, L, C = x.shape

        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 2, C).permute(2, 0, 1, 3)

        x1 = self.attns[0](qkv[:, :, :, :C // 2])
        x2 = self.attns[1](qkv[:, :, :, C // 2:])
        attened_x = torch.cat([x1, x2], dim=2)

        attened_x = self.proj(attened_x)#去掉？
        x = x + self.drop_path(attened_x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Cross_CSWinBlock(nn.Module):

    def __init__(self, dim, num_heads, patches_resolution=None,
                 split_size=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.patches_resolution = patches_resolution
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim * 2)

        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            LePEAttention(
                dim // 2, resolution=self.patches_resolution, idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        """
        x: B, H*W, C
        """
        b, c, h, w = x1.shape
        x1 = x1.view(b, c, -1).permute(0, 2, 1)
        B, L, C = x2.shape

        img_c = self.norm1(x1)#(1,256,256)
        img_t = self.norm1(x2)#(1,256,256)
        qkv1 = torch.stack((img_c, img_t), dim=0)#torch.stack()将img_c和img_t张量沿着第0维进行堆叠(2,1,256,256)

        x1 = self.attns[0](qkv1[:, :, :, :C // 2])
        x2 = self.attns[1](qkv1[:, :, :, C // 2:])
        attened_x = torch.cat([x1, x2], dim=2)

        attened_x = self.proj(attened_x)
        x = img_c + self.drop_path(attened_x)

        qkv2 = torch.stack((img_t, x), dim=0)
        x1 = self.attns[0](qkv2[:, :, :, :C // 2])
        x2 = self.attns[1](qkv2[:, :, :, C // 2:])
        attened_x = torch.cat([x1, x2], dim=2)

        attened_x = self.proj(attened_x)
        x = img_t + self.drop_path(attened_x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, last=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        if last == False:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        else:
            self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, image_size=128):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        self.image_down = nn.Conv2d(64, 64, kernel_size=2, stride=2)


        dpr = [x.item() for x in torch.linspace(0, 0.4, 4)]

        # --------------------下采样编码器---------------------
        # self.C_down1 = Down(64, 128)
        # self.TF_down1 = Self_CSWinBlock(64, num_heads=4, patches_resolution=image_size, mlp_ratio=4, qkv_bias=True,
        #                                 qk_scale=None, drop=0, attn_drop=0,
        #                                 drop_path=dpr[0], norm_layer=nn.LayerNorm)
        # self.T_down1 = PatchMerging(image_size, dim=64, norm_layer=nn.LayerNorm, last=False)
        # self.Cross_ATT1_down = Cross_CSWinBlock(128, num_heads=4, patches_resolution=image_size // 2, mlp_ratio=4,
        #                                         qkv_bias=True, qk_scale=None, drop=0,
        #                                         attn_drop=0,
        #                                         drop_path=dpr[0], norm_layer=nn.LayerNorm)
        #
        # self.C_down2 = Down(128, 256)
        # self.TF_down2 = Self_CSWinBlock(128, num_heads=8, patches_resolution=image_size // 2, mlp_ratio=4,
        #                                 qkv_bias=True, qk_scale=None, drop=0,
        #                                 attn_drop=0,
        #                                 drop_path=dpr[1], norm_layer=nn.LayerNorm)
        # self.T_down2 = PatchMerging(image_size // 2, dim=128, norm_layer=nn.LayerNorm, last=False)
        # self.Cross_ATT2_down = Cross_CSWinBlock(256, num_heads=8, patches_resolution=image_size // 4, mlp_ratio=4,
        #                                         qkv_bias=True, qk_scale=None, drop=0,
        #                                         attn_drop=0,
        #                                         drop_path=dpr[1], norm_layer=nn.LayerNorm)
        #
        # self.C_down3 = Down(256, 512)
        # self.TF_down3 = Self_CSWinBlock(256, num_heads=16, patches_resolution=image_size // 4, mlp_ratio=4,
        #                                 qkv_bias=True, qk_scale=None, drop=0,
        #                                 attn_drop=0,
        #                                 drop_path=dpr[2], norm_layer=nn.LayerNorm)
        # self.T_down3 = PatchMerging(image_size // 4, dim=256, norm_layer=nn.LayerNorm, last=False)
        # self.Cross_ATT3_down = Cross_CSWinBlock(512, num_heads=16, patches_resolution=image_size // 8, mlp_ratio=4,
        #                                         qkv_bias=True, qk_scale=None, drop=0,
        #                                         attn_drop=0,
        #                                         drop_path=dpr[2], norm_layer=nn.LayerNorm)
        #
        # self.C_down4 = Down(512, 512)
        # self.TF_down4 = Self_CSWinBlock(512, num_heads=32, patches_resolution=image_size // 8, mlp_ratio=4,
        #                                 qkv_bias=True, qk_scale=None, drop=0,
        #                                 attn_drop=0,
        #                                 drop_path=dpr[3], norm_layer=nn.LayerNorm)
        # self.T_down4 = PatchMerging(image_size // 8, dim=512, norm_layer=nn.LayerNorm, last=True)
        # self.Cross_ATT4_down = Cross_CSWinBlock(512, num_heads=32, patches_resolution=image_size // 16, mlp_ratio=4,
        #                                         qkv_bias=True, qk_scale=None, drop=0,
        #                                         attn_drop=0,
        #                                         drop_path=dpr[3], norm_layer=nn.LayerNorm)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.eca1 = ECAAttention(kernel_size=3)
        self.eca2 = ECAAttention(kernel_size=3)
        self.eca3 = ECAAttention(kernel_size=3)
        self.eca4 = ECAAttention(kernel_size=3)

        # --------------------上采样解码器---------------------
        self.C_up4 = Up(1024, 256, bilinear)
        self.TF_up4 = Self_CSWinBlock(256, num_heads=32, mlp_ratio=4, patches_resolution=image_size // 8,
                                      qkv_bias=True, qk_scale=None, drop=0, attn_drop=0,
                                      drop_path=dpr[3], norm_layer=nn.LayerNorm)
        # self.T_up4 = PatchExpand(image_size // 16, dim=512, dim_scale=2, norm_layer=nn.LayerNorm, last=True)
        self.Cross_ATT4_up = Cross_CSWinBlock(256, num_heads=32, patches_resolution=image_size // 8, mlp_ratio=4,
                                              qkv_bias=True, qk_scale=None, drop=0,
                                              attn_drop=0,
                                              drop_path=dpr[3], norm_layer=nn.LayerNorm)

        self.C_up3 = Up(512, 128, bilinear)
        self.TF_up3 = Self_CSWinBlock(128, num_heads=16, mlp_ratio=4, patches_resolution=image_size // 4, qkv_bias=True,
                                      qk_scale=None, drop=0, attn_drop=0,
                                      drop_path=dpr[2], norm_layer=nn.LayerNorm)
        # self.T_up3 = PatchExpand(image_size // 8, dim=256, dim_scale=2, norm_layer=nn.LayerNorm, last=False)

        self.Cross_ATT3_up = Cross_CSWinBlock(128, num_heads=16, patches_resolution=image_size // 4, mlp_ratio=4,
                                              qkv_bias=True, qk_scale=None, drop=0,
                                              attn_drop=0,
                                              drop_path=dpr[2], norm_layer=nn.LayerNorm)

        self.C_up2 = Up(256, 64, bilinear)
        self.TF_up2 = Self_CSWinBlock(64, num_heads=8, mlp_ratio=4, patches_resolution=image_size // 2, qkv_bias=True,
                                      qk_scale=None, drop=0, attn_drop=0,
                                      drop_path=dpr[1], norm_layer=nn.LayerNorm)
        # self.T_up2 = PatchExpand(image_size // 4, dim=128, dim_scale=2, norm_layer=nn.LayerNorm, last=False)
        self.Cross_ATT2_up = Cross_CSWinBlock(64, num_heads=8, patches_resolution=image_size // 2, mlp_ratio=4,
                                              qkv_bias=True, qk_scale=None, drop=0,
                                              attn_drop=0,
                                              drop_path=dpr[1], norm_layer=nn.LayerNorm)

        self.C_up1 = Up(128, 64, bilinear)
        self.TF_up1 = Self_CSWinBlock(64, num_heads=4, patches_resolution=image_size, mlp_ratio=4, qkv_bias=True,
                                      qk_scale=None, drop=0, attn_drop=0,
                                      drop_path=dpr[0], norm_layer=nn.LayerNorm)
        # self.T_up1 = PatchExpand(image_size // 2, dim=64, dim_scale=2, norm_layer=nn.LayerNorm, last=False)
        self.Cross_ATT1_up = Cross_CSWinBlock(64, num_heads=4, patches_resolution=image_size, mlp_ratio=4,
                                              qkv_bias=True, qk_scale=None, drop=0,
                                              attn_drop=0,
                                              drop_path=dpr[0], norm_layer=nn.LayerNorm)

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv4 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # self.up4 = Up(1024,256,bilinear)
        # self.up3 = Up(512,128,bilinear)
        # self.up2 = Up(256,64,bilinear)
        # self.up1 = Up(128,64,bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.image_down(x1)

        # x1_C = self.C_down1(x1)
        # x1_T = self.T_down1(self.TF_down1(x1))
        # x2 = self.Cross_ATT1_down(x1_C, x1_T)
        #
        # x2_C = self.C_down2(x2)
        # x2_T = self.T_down2(self.TF_down2(x2))
        # x3 = self.Cross_ATT2_down(x2_C, x2_T)
        #
        # x3_C = self.C_down3(x3)
        # x3_T = self.T_down3(self.TF_down3(x3))
        # x4 = self.Cross_ATT3_down(x3_C, x3_T)
        #
        # x4_C = self.C_down4(x4)
        # x4_T = self.T_down4(self.TF_down4(x4))
        # x5 = self.Cross_ATT4_down(x4_C, x4_T)

        x2_ = self.down1(x1)
        x2 = self.eca1(x2_)
        x3_ = self.down2(x2_)
        x3 = self.eca2(x3_)
        x4_ = self.down3(x3_)
        x4 = self.eca3(x4_)
        x5_ = self.down4(x4_)
        x5 = self.eca4(x5_)

        x_C = self.C_up4(x5, x4)
        x = torch.cat([self.up_sample(x5), x4], dim=1)
        x = self.conv4(x)
        b, c, h, w = x.shape
        x_T = (self.TF_up4(x).permute(0, 2, 1).view(b, c, h, w)).view(b, c, -1).permute(0, 2, 1)
        x = self.Cross_ATT4_up(x_C, x_T)

        x_C = self.C_up3(x, x3)
        x = torch.cat([self.up_sample(x), x3], dim=1)
        x = self.conv3(x)
        b, c, h, w = x.shape
        x_T = (self.TF_up3(x).permute(0, 2, 1).view(b, c, h, w)).view(b, c, -1).permute(0, 2, 1)
        x = self.Cross_ATT3_up(x_C, x_T)

        x_C = self.C_up2(x, x2)
        x = torch.cat([self.up_sample(x), x2], dim=1)
        x = self.conv2(x)
        b, c, h, w = x.shape
        x_T = (self.TF_up2(x).permute(0, 2, 1).view(b, c, h, w)).view(b, c, -1).permute(0, 2, 1)
        x = self.Cross_ATT2_up(x_C, x_T)

        x_C = self.C_up1(x, x1)
        x_F = torch.cat([self.up_sample(x), x1], dim=1)
        x = self.conv1(x_F)
        b, c, h, w = x.shape
        x_T = (self.TF_up1(x).permute(0, 2, 1).view(b, c, h, w)).view(b, c, -1).permute(0, 2, 1)
        x = self.Cross_ATT1_up(x_C, x_T)

        # x = self.up4(x5, x4)
        # x = self.up3(x, x3)
        # x = self.up2(x, x2)
        # x = self.up1(x, x1)

        x = self.up_sample(x)

        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    import torch as t

    print('-----' * 5)
    rgb = t.randn(1, 3, 256, 256)

    net = UNet(3, 1)
    print('Trainable model parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad == True) / 1e6,
          'M')
    out = net(rgb)


    print(out.shape)
