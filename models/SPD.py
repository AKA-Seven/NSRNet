import torch
import torch.nn as nn
import torch.nn.functional as F


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),  # ← 增加两层
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fcn(x)



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 输入通道为6（图像 + noise map），输出通道为64
        self.inc = nn.Sequential(
            single_conv(6, 64),     # [B, 6, H, W] → [B, 64, H, W]
            single_conv(64, 64)     # [B, 64, H, W] → [B, 64, H, W]
        )

        # 下采样模块1：H, W 减半
        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),   # [B, 64, H/2, W/2] → [B, 128, H/2, W/2]
            single_conv(128, 128),
            single_conv(128, 128)
        )

        # 下采样模块2
        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),  # [B, 128, H/4, W/4] → [B, 256, H/4, W/4]
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        # 上采样模块1：通道 256 → 128，H,W * 2
        self.up1 = up(256)  # 输入 x1 是 [B, 256, H/4, W/4]，输出 [B, 128, H/2, W/2]
        self.conv3 = nn.Sequential(
            single_conv(128, 128),  # 输入来自 skip + up 的输出：[B, 128, H/2, W/2]
            single_conv(128, 128),
            single_conv(128, 128)
        )

        # 上采样模块2：通道 128 → 64
        self.up2 = up(128)  # 输出 [B, 64, H, W]
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        # 最后一层 1x1 卷积，压缩为3通道输出
        self.outc = outconv(64, 3)  # [B, 64, H, W] → [B, 3, H, W]

    def forward(self, x):
        inx = self.inc(x)              # [B, 6, H, W] → [B, 64, H, W]

        down1 = self.down1(inx)        # → [B, 64, H/2, W/2]
        conv1 = self.conv1(down1)      # → [B, 128, H/2, W/2]

        down2 = self.down2(conv1)      # → [B, 128, H/4, W/4]
        conv2 = self.conv2(down2)      # → [B, 256, H/4, W/4]

        up1 = self.up1(conv2, conv1)   # → [B, 128, H/2, W/2]
        conv3 = self.conv3(up1)        # → [B, 128, H/2, W/2]

        up2 = self.up2(conv3, inx)     # → [B, 64, H, W]
        conv4 = self.conv4(up2)        # → [B, 64, H, W]

        out = self.outc(conv4)         # → [B, 3, H, W]
        return out



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fcn = FCN()
        self.unet = UNet()
    
    def forward(self, x):
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x
        return noise_level, out