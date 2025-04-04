import torch.nn as nn
import torch
import os
from net.init_weights import init_weights
from net.layers import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
    ResidualConv_WDDA,
    EUCB,

)

#=================================================================
import torch.nn as nn
from torch.nn import functional as F
import torch




import torch
import torch.nn as nn


class Upsample1_(nn.Module):
    def __init__(self, in_channels, scale=2):
        super(Upsample1_, self).__init__()

        # 上采样层，使用双线性插值
        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

        # 1x1 卷积层，用来减少通道数，输出通道数是输入通道数的一半
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, x):
        # 先进行上采样
        x = self.upsample(x)
        # 然后使用 1x1 卷积减少通道数
        x = self.conv(x)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
class CAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True, groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)
        out_conv = out_conv.squeeze(2)

        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv
        return output

class CAFMFusion(nn.Module):
    def __init__(self, dim):
        super(CAFMFusion, self).__init__()
        self.CAFM = CAFM(dim)
        self.PixelAttention = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        pattn1 = self.CAFM(initial)
        pattn2 = self.sigmoid(self.PixelAttention(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

#=============================================================================
class L_WDDA_Net(nn.Module):
    def __init__(self, args):
        super(L_WDDA_Net, self).__init__()
        self.args = args
        in_channels = 3
        n_classes = 2
        filters=[16, 32, 64, 128, 256]

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )


        self.residual_conv1 = ResidualConv_WDDA(filters[0], filters[1], 2, 1)


        self.residual_conv2 = ResidualConv_WDDA(filters[1], filters[2], 2, 1)


        self.residual_conv3 = ResidualConv_WDDA(filters[2], filters[3], 2, 1)

        self.residual_conv4 = ResidualConv_WDDA(filters[3], filters[4], 2, 1)




        self.upsample1 = EUCB(filters[4],filters[3])
        self.cafm1 = CAFMFusion(filters[3])

        self.up_residual_conv1 = ResidualConv(filters[3] + filters[3], filters[3], 1, 1)


        self.upsample2 = EUCB(filters[3],filters[2])
        self.cafm2 = CAFMFusion(filters[2])
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[2], filters[2], 1, 1)


        self.upsample3 = EUCB(filters[2],filters[1])
        self.cafm3 = CAFMFusion(filters[1])
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[1], filters[1], 1, 1)


        self.upsample4 = EUCB(filters[1],filters[0])
        self.cafm4 = CAFMFusion(filters[0])
        self.up_residual_conv4 = ResidualConv(filters[0] + filters[0], filters[0], 1, 1)




        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], n_classes, 1))

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)


        x2 = self.residual_conv1(x1)


        x3 = self.residual_conv2(x2)


        x4 = self.residual_conv3(x3)


        x5 = self.residual_conv4(x4)


        x6 = self.upsample1(x5)
        x6 = self.cafm1(x4, x6)
        x6 = torch.cat([x6, x4], dim=1)
        x6 = self.up_residual_conv1(x6)


        x7 = self.upsample2(x6)
        x7 = self.cafm2(x3, x7)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.up_residual_conv2(x7)


        x8 = self.upsample3(x7)
        x8 = self.cafm3(x2, x8)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.up_residual_conv3(x8)


        x9 = self.upsample4(x8)
        x9 = self.cafm4(x1, x9)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = self.up_residual_conv4(x9)


        out = self.output_layer(x9)

        return out



