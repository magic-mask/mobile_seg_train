# model/ultralight_unet.py

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UltraLightUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_c=32,
    ):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_c)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_c, base_c * 2),
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_c * 2, base_c * 4),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_c * 4, base_c * 8),
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_c * 8, base_c * 16),
        )

        self.up1 = nn.ConvTranspose2d(
            base_c * 16,
            base_c * 8,
            kernel_size=2,
            stride=2,
        )
        self.conv_up1 = DoubleConv(
            base_c * 16,
            base_c * 8,
        )

        self.up2 = nn.ConvTranspose2d(
            base_c * 8,
            base_c * 4,
            kernel_size=2,
            stride=2,
        )
        self.conv_up2 = DoubleConv(
            base_c * 8,
            base_c * 4,
        )

        self.up3 = nn.ConvTranspose2d(
            base_c * 4,
            base_c * 2,
            kernel_size=2,
            stride=2,
        )
        self.conv_up3 = DoubleConv(
            base_c * 4,
            base_c * 2,
        )

        self.up4 = nn.ConvTranspose2d(
            base_c * 2,
            base_c,
            kernel_size=2,
            stride=2,
        )
        self.conv_up4 = DoubleConv(
            base_c * 2,
            base_c,
        )

        self.outc = nn.Conv2d(
            base_c,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        return self.outc(x)