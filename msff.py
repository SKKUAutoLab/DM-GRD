import torch.nn as nn
from non_local import NONLocalBlock2D
import torch

class MSFFBlock(nn.Module):
    def __init__(self, in_channel):
        super(MSFFBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
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
        # non-local attention
        f1_k = self.blk1(f1) # [8, 64, 64, 64]
        f2_k = self.blk2(f2) # [8, 128, 32, 32]
        f3_k = self.blk3(f3) # [8, 256, 16, 16]
        f2_f = f2_k + self.upconv32(f3_k) # [8, 128, 32, 32]
        f1_f = f1_k + self.upconv21(f2_f) # [8, 64, 64, 64]
        # spatial attention
        m3 = self.sam(f3) # [8, 256, 16, 16] -> [8, 1, 16, 16]
        m2 = self.sam(f2) * self.upsample(m3) # [8, 1, 32, 32]
        m1 = self.sam(f1) * self.upsample(m2) # [8, 1, 64, 64]
        f1_out = f1_f * m1 # [8, 64, 64, 64]
        f2_out = f2_f * m2 # [8, 128, 32, 32]
        f3_out = f3_k * m3 # [8, 256, 16, 16]
        return [f1_out, f2_out, f3_out]