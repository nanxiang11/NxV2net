import torch
import torch.nn as nn
from BaseConv import DSConv
from BaseConv.MCFA import MCFA
from torchsummary import summary


class DS_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DS_conv, self).__init__()
        self.layer = nn.Sequential(
            DSConv.DSConv_pro(in_channels, out_channels, kernel_size, device="cuda").cuda(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        ).to('cuda')
 
    def forward(self, x):
        return self.layer(x)


class BaseUnet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(BaseUnet, self).__init__()

        self.conv0_0 = MCFA(in_channels, out_channels)

        self.conv1_0 = DS_conv(in_channels=out_channels, out_channels=out_channels*2, kernel_size=kernel_size)

        self.maxpool = nn.MaxPool2d(2)

        self.upsample = nn.ConvTranspose2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        self.conv0_1 = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)


    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.maxpool(x0_0))

        x0_1 = torch.cat((x0_0, self.upsample(x1_0)), dim=1)
        x0_1 = self.conv0_1(x0_1)

        return x0_1

class BaseUnetDS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BaseUnetDS, self).__init__()
        self.conv0_0 = DS_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv1_0 = DS_conv(in_channels=out_channels, out_channels=out_channels*2, kernel_size=kernel_size)

        self.maxpool = nn.MaxPool2d(2)

        self.upsample = nn.ConvTranspose2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        self.conv0_1 = DS_conv(in_channels=out_channels*2, out_channels=out_channels, kernel_size=kernel_size)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.maxpool(x0_0))

        x0_1 = torch.cat((x0_0, self.upsample(x1_0)), dim=1)
        x0_1 = self.conv0_1(x0_1)

        return x0_1


class V2net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(V2net, self).__init__()

        self.NX0_0 = BaseUnet(in_channels, 64, kernel_size=3)
        self.NX1_0 = BaseUnet(64, 128, kernel_size=3)
        self.NX2_0 = BaseUnet(128, 256, kernel_size=3)
        self.NX3_0 = BaseUnetDS(256, 512, kernel_size=3)


        self.maxpool = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(4)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.C0_2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)
        self.C3_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)
        self.C2_0 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)


        self.upsample2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        self.conv2_1 = BaseUnet(256 * 2, 256, kernel_size=3)
        self.conv1_1 = BaseUnet(128 * 2, 128, kernel_size=3)
        self.conv0_1 = BaseUnet(64 * 2, 64, kernel_size=3)

        self.sg = torch.nn.Sigmoid()
        self.pre = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )


    def forward(self, x):
        x0_0 = self.NX0_0(x)
        x1_0 = self.NX1_0(self.maxpool(x0_0))
        x2_0 = self.NX2_0(self.maxpool(x1_0))
        x3_0 = self.NX3_0(self.maxpool(x2_0))


        x2_1 = torch.cat((self.upsample2_1(x3_0), x2_0 + self.C0_2(self.maxpool4(x0_0))), dim=1)
        x2_1 = self.conv2_1(x2_1)

        x1_1 = torch.cat((self.upsample1_1(x2_1) + self.C3_1(self.upsample(x3_0)), x1_0), dim=1)
        x1_1 = self.conv1_1(x1_1)

        x0_1 = torch.cat((self.upsample0_1(x1_1) + self.C2_0(self.upsample(x2_1)), x0_0), dim=1)
        x0_1 = self.conv0_1(x0_1)

        return self.sg(self.pre(x0_1))






input = (3, 224, 224)
net = V2net(3,  3).cuda()
summary(net, input)