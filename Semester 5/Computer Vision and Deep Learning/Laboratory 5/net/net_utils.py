import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size, stride)
        self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # expand image
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        # crop image
        # diff_y = x1.size(2) - x2.size(2)
        # diff_x = x1.size(3) - x2.size(3)
        # x1 = x1[:, :, diff_y // 2:x1.size(2) - diff_y // 2, diff_x // 2:x1.size(3) - diff_x // 2]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride),
            DoubleConvolution(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv(x)
