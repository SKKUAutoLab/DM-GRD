import torch.nn as nn
# from coordatt import CoordAtt
from non_local import NONLocalBlock2D
import torch

class MSFFBlock(nn.Module):
    def __init__(self, in_channel):
        super(MSFFBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        # self.attn = CoordAtt(in_channel, in_channel)
        self.attn = NONLocalBlock2D(in_channel)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1), nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, stride=1, padding=1))

    def forward(self, x): # [8, 128, 64, 64]
        x_conv = self.conv1(x) # [8, 128, 64, 64]
        x_att = self.attn(x) # [8, 128, 64, 64]
        x = x_conv * x_att # [8, 128, 64, 64]
        x = self.conv2(x) # [8, 64, 64, 64]
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class MSFF(nn.Module):
    def __init__(self):
        super(MSFF, self).__init__()
        self.blk1 = MSFFBlock(512)
        self.blk2 = MSFFBlock(1024)
        self.blk3 = MSFFBlock(2048)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv32 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1))
        self.upconv21 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.sam = SpatialAttention()

    def forward(self, features):
        f1, f2, f3 = features # [8, 128, 64, 64], [8, 256, 32, 32], [8, 512, 16, 16]
        f1_k = self.blk1(f1) # [8, 64, 64, 64]
        f2_k = self.blk2(f2) # [8, 128, 32, 32]
        f3_k = self.blk3(f3) # [8, 256, 16, 16]
        f2_f = f2_k + self.upconv32(f3_k) # [8, 128, 32, 32]
        f1_f = f1_k + self.upconv21(f2_f) # [8, 64, 64, 64]
        # spatial-attention module
        m3 = self.sam(f3) # [8, 256, 16, 16] -> [8, 1, 16, 16]
        m2 = self.sam(f2) * self.upsample(m3) # [8, 1, 32, 32]
        m1 = self.sam(f1) * self.upsample(m2) # [8, 1, 64, 64]
        f1_out = f1_f * m1 # [8, 64, 64, 64]
        f2_out = f2_f * m2 # [8, 128, 32, 32]
        f3_out = f3_k * m3 # [8, 256, 16, 16]
        return [f1_out, f2_out, f3_out]

# import torch
# class MSFFBlock(nn.Module):
#     def __init__(self, in_channel):
#         super(MSFFBlock, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel // 4, kernel_size=3, stride=1, dilation=1, padding=1), nn.ReLU(), nn.BatchNorm2d(in_channel // 4))
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channel, in_channel // 4, kernel_size=3, stride=1, dilation=2, padding=2), nn.ReLU(), nn.BatchNorm2d(in_channel // 4))
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channel, in_channel // 4, kernel_size=3, stride=1, dilation=4, padding=4), nn.ReLU(), nn.BatchNorm2d(in_channel // 4))
#         self.conv4 = nn.Sequential(nn.Conv2d(in_channel, in_channel // 4, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU())
#         self.conv5 = nn.Sequential(nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(in_channel // 2))
#         self.attn = CoordAtt(in_channel // 4, in_channel // 4)
#         # self.attn = NONLocalBlock2D(in_channel // 4)
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#         out2 = self.conv2(x)
#         out3 = self.conv3(x)
#         out_d = torch.cat((out1, out2, out3), dim=1)
#         out = self.conv4(x)
#         out = self.attn(out)
#         out = torch.cat((out_d, out), dim=1)
#         out = self.conv5(out)
#         return out
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_result, _ = torch.max(x, dim=1, keepdim=True)
#         avg_result = torch.mean(x, dim=1, keepdim=True)
#         result = torch.cat([max_result, avg_result], 1)
#         output = self.conv(result)
#         output = self.sigmoid(output)
#         return output
#
# class MSFF(nn.Module):
#     def __init__(self):
#         super(MSFF, self).__init__()
#         self.block1 = MSFFBlock(128)
#         self.block2 = MSFFBlock(256)
#         self.block3 = MSFFBlock(512)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.upconv32 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
#         self.upconv21 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
#         self.sam = SpatialAttention()
#
#     def forward(self, features):
#         f1, f2, f3 = features # [8, 128, 64, 64], [8, 256, 32, 32], [8, 512, 16, 16]
#         f1_k = self.block1(f1) # [8, 64, 64, 64]
#         f2_k = self.block2(f2) # [8, 128, 32, 32]
#         f3_k = self.block3(f3) # [8, 256, 16, 16]
#         f2_f = f2_k + self.upconv32(f3_k) # [8, 128, 32, 32]
#         f1_f = f1_k + self.upconv21(f2_f) # [8, 64, 64, 64]
#         # spatial-attention module
#         m3 = self.sam(f3)
#         m2 = self.sam(f2) * self.upsample(m3)
#         m1 = self.sam(f1) * self.upsample(m2)
#         f1_out = f1_f * m1  # [8, 64, 64, 64]
#         f2_out = f2_f * m2  # [8, 128, 32, 32]
#         f3_out = f3_k * m3 # [8, 256, 16, 16]
#         return [f1_out, f2_out, f3_out]

# import torch
# import torch.nn as nn
#
# class FFCSE_block(nn.Module):
#     def __init__(self, channels, ratio_g):
#         super(FFCSE_block, self).__init__()
#         in_cg = int(channels * ratio_g)
#         in_cl = channels - in_cg
#         r = 16
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv_a2l = None if in_cl == 0 else nn.Conv2d(channels // r, in_cl, kernel_size=1, bias=True)
#         self.conv_a2g = None if in_cg == 0 else nn.Conv2d(channels // r, in_cg, kernel_size=1, bias=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = x if type(x) is tuple else (x, 0)
#         id_l, id_g = x
#         x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
#         x = self.avgpool(x)
#         x = self.relu1(self.conv1(x))
#         x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
#         x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
#         return x_l, x_g
#
# class FourierUnit(nn.Module):
#     def __init__(self, in_channels, out_channels, groups=1):
#         super(FourierUnit, self).__init__()
#         self.groups = groups
#         self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
#         self.bn = torch.nn.BatchNorm2d(out_channels * 2)
#         self.relu = torch.nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         batch, c, h, w = x.size()
#         ffted = torch.fft.rfftn(x, s=(h, w), dim=(2, 3), norm='ortho')
#         ffted = torch.cat([ffted.real, ffted.imag], dim=1)
#         ffted = self.conv_layer(ffted)
#         ffted = self.relu(self.bn(ffted))
#         ffted = torch.tensor_split(ffted, 2, dim=1)
#         ffted = torch.complex(ffted[0], ffted[1])
#         output = torch.fft.irfftn(ffted, s=(h, w), dim=(2, 3), norm='ortho')
#         return output
#
# class SpectralTransformer(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
#         super(SpectralTransformer, self).__init__()
#         self.enable_lfu = enable_lfu
#         if stride == 2:
#             self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
#         else:
#             self.downsample = nn.Identity()
#         self.stride = stride
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False), nn.BatchNorm2d(out_channels // 2), nn.ReLU(inplace=True))
#         self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
#         if self.enable_lfu:
#             self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
#         self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)
#
#     def forward(self, x):
#         x = self.downsample(x) # [640, 512, 32]
#         x = self.conv1(x) # [640, 128, 32]
#         output = self.fu(x) # [640, 128, 32]
#         if self.enable_lfu:
#             n, c, h, w = x.shape
#             split_no = 2
#             split_s_h = h // split_no
#             split_s_w = w // split_no
#             xs = torch.cat(torch.split(x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
#             xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
#             xs = self.lfu(xs)
#             xs = xs.repeat(1, 1, split_no, split_no).contiguous()
#         else:
#             xs = 0
#         output = self.conv2(x + output + xs) # [640, 128, 32]
#         return output
#
# class FFC(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1, groups=1, bias=False, enable_lfu=True):
#         super(FFC, self).__init__()
#         assert stride == 1 or stride == 2, "Stride should be 1 or 2."
#         self.stride = stride
#         in_cg = int(in_channels * ratio_gin) # input channel global
#         in_cl = in_channels - in_cg # input channel local
#         out_cg = int(out_channels * ratio_gout) # output channel global
#         out_cl = out_channels - out_cg # output channel local
#         self.ratio_gin = ratio_gin
#         self.ratio_gout = ratio_gout
#         module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
#         self.l2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias)
#         module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
#         self.l2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias)
#         module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
#         self.g2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias)
#         module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransformer
#         self.g2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)
#
#     def forward(self, x):
#         x_l, x_g = x if type(x) is tuple else (x, 0)
#         y_l, y_g = 0, 0
#         if self.ratio_gout != 1:
#             y_l = self.l2l(x_l) + self.g2l(x_g)
#         if self.ratio_gout != 0:
#             y_g = self.g2g(x_g) + self.l2g(x_l)
#         return y_l, y_g
#
# class FFC_BN_ACT(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity, enable_lfu=True):
#         super(FFC_BN_ACT, self).__init__()
#         self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias, enable_lfu)
#         lnorm = nn.Identity if ratio_gout == 1 else norm_layer
#         gnorm = nn.Identity if ratio_gout == 0 else norm_layer
#         self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
#         self.bn_g = gnorm(int(out_channels * ratio_gout))
#         lact = nn.Identity if ratio_gout == 1 else activation_layer
#         gact = nn.Identity if ratio_gout == 0 else activation_layer
#         self.act_l = lact(inplace=True)
#         self.act_g = gact(inplace=True)
#
#     def forward(self, x):
#         x_l, x_g = self.ffc(x)
#         x_l = self.act_l(self.bn_l(x_l))
#         x_g = self.act_g(self.bn_g(x_g))
#         return x_l, x_g
#
# class FFC_Block(nn.Module):
#     def __init__(self, in_channel):
#         super(FFC_Block, self).__init__()
#         # self.attn = NONLocalBlock2D(in_channel // 2, sub_sample=False, bn_layer=True)
#         self.attn = CoordAtt(in_channel // 2, in_channel // 2)
#         self.conv = nn.Sequential(FFC_BN_ACT(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1, ratio_gin=0, ratio_gout=0.5),
#                                   FFC_BN_ACT(in_channel // 2, in_channel // 2, kernel_size=3, stride=1, padding=1, ratio_gin=0.5, ratio_gout=0))
#         self.se_block = FFCSE_block(in_channel // 2, 0)
#
#     def forward(self, x): # [8, 128, 64, 64]
#         x_conv = self.conv(x) # [8, 64, 64, 64]
#         x_l, x_g = self.se_block(x_conv)
#         x_att = self.attn(x_l) # [8, 64, 64, 64]
#         return x_att
#
# class UpsampleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2):
#         super().__init__()
#         self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
#
#     def forward(self, x):
#         x = self.upsample(x)
#         x = self.conv(x)
#         return x
#
# class MSFF(nn.Module):
#     def __init__(self, channels=(64, 128, 256)):
#         super().__init__()
#         # ffc block
#         self.block1 = FFC_Block(128)
#         self.block2 = FFC_Block(256)
#         self.block3 = FFC_Block(512)
#         # out1 = out1 + out21 + out31
#         self.layer_21 = UpsampleConv(channels[1], channels[0], scale_factor=2)
#         self.layer_31 = UpsampleConv(channels[2], channels[0], scale_factor=4)
#         # out2 = out12 + out2 + out32
#         self.layer_12 = nn.Conv2d(channels[0], channels[1], stride=2, kernel_size=3, padding=1, groups=channels[0])
#         self.layer_32 = UpsampleConv(channels[2], channels[1], scale_factor=2)
#         # out3 = out13 + out23 + out3
#         self.layer_13 = nn.Conv2d(channels[0], channels[2], stride=4, kernel_size=5, padding=2, groups=channels[0])
#         self.layer_23 = nn.Conv2d(channels[1], channels[2], stride=2, kernel_size=3, padding=1, groups=channels[1])
#
#     def forward(self, features):
#         f1, f2, f3 = features # [8, 128, 64, 64], [8, 256, 32, 32], [8, 512, 16, 16]
#         f1_ffc = self.block1(f1) # [8, 64, 64, 64]
#         f2_ffc = self.block2(f2) # [8, 128, 32, 32]
#         f3_ffc = self.block3(f3) # [8, 256, 16, 16]
#         out1 = f1_ffc + self.layer_21(f2_ffc) + self.layer_31(f3_ffc) # [8, 64, 64, 64]
#         out2 = self.layer_12(f1_ffc) + f2_ffc + self.layer_32(f3_ffc) # [8, 128, 32, 32]
#         out3 = self.layer_13(f1_ffc) + self.layer_23(f2_ffc) + f3_ffc # [8, 256, 16, 16]
#         return [out1, out2, out3]
#
# if __name__ == '__main__':
#     x = [torch.randn(8, 128, 64, 64), torch.rand(8, 256, 32, 32), torch.rand(8, 512, 16, 16)]
#     msff = MSFF(channels=(64, 128, 256))
#     out1, out2, out3 = msff(x)
#     print(out1.shape)
#     print(out2.shape)
#     print(out3.shape)