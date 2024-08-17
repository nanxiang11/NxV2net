import torch
from torch import nn


class MCFA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.A = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel)
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.silu = nn.SiLU()

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.A(x) * x1
        y = self.silu(x2 + x3)
        return y


if __name__ == '__main__':
    input = torch.randn(3, 3, 7, 7)
    pna = MCFA(3, 64)
    output = pna(input)
    print(output.shape)
