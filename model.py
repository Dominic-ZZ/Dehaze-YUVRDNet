import torch.nn as nn
import torch.nn.functional as F
import torch
from resnet import ResNet18


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvBlock(nn.Module):
    """implement conv+ReLU two times"""

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=middle_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )
        conv_relu.append(nn.ReLU())
        conv_relu.append(
            nn.Conv2d(
                in_channels=middle_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)

    def forward(self, x):
        out = self.conv_ReLU(x)
        return out


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(
            nChannels,
            growthRate,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer=3, growthRate=32):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(
            nChannels_, nChannels, kernel_size=1, padding=0, bias=False
        )

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)

        out = out + x
        return out


def tensor2y_uv(tensor):
    # print(tensor.shape)
    y, u, v = torch.split(tensor, 1, dim=1)
    uv = torch.cat((u, v), dim=1)
    return y, uv


class YUV_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.left_conv_1 = ConvBlock(in_channels=1, middle_channels=64, out_channels=64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = ConvBlock(
            in_channels=64, middle_channels=128, out_channels=128
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = ConvBlock(
            in_channels=128, middle_channels=256, out_channels=256
        )
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_4 = ConvBlock(
            in_channels=256, middle_channels=512, out_channels=512
        )
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_5 = ConvBlock(
            in_channels=512, middle_channels=1024, out_channels=1024
        )

        # self.deconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv_1 = nn.Conv2d(
            in_channels=1024, out_channels=512, kernel_size=3, padding=1, stride=1
        )
        self.right_conv_1 = ConvBlock(
            in_channels=512 + 256, middle_channels=512, out_channels=256
        )

        # self.deconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.upconv_2 = nn.Conv2d(
            in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1
        )
        self.right_conv_2 = ConvBlock(
            in_channels=640, middle_channels=256 + 128, out_channels=128
        )

        # self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2 ,output_padding=1)
        self.upconv_3 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1
        )
        self.right_conv_3 = ConvBlock(
            in_channels=320, middle_channels=128 * 2, out_channels=64
        )

        # self.deconv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.upconv_4 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1
        )
        self.right_conv_4 = ConvBlock(
            in_channels=128, middle_channels=128, out_channels=64
        )

        self.right_conv_5 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0
        )

        self.tanh = nn.Tanh()

        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.upsample_2 = nn.Upsample(scale_factor=2)
        self.upsample_3 = nn.Upsample(scale_factor=2)
        self.upsample_4 = nn.Upsample(scale_factor=2)

        self.rdb_conv_1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=2
        )
        self.rdb_conv_2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2
        )
        self.rdb_conv_3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2
        )

        self.rdb_1 = RDB(nChannels=64)
        self.rdb_2 = RDB(nChannels=128)
        self.rdb_3 = RDB(nChannels=256)
        self.rdb_4 = RDB(nChannels=512)

        self.BN_1 = nn.BatchNorm2d(64)
        self.BN_2 = nn.BatchNorm2d(128)
        self.BN_3 = nn.BatchNorm2d(256)

        self.se1 = SELayer(512 + 256)
        self.se2 = SELayer(640)
        self.se3 = SELayer(320)
        self.se4 = SELayer(66)
        self.sa1 = SALayer()
        self.sa2 = SALayer()
        self.sa3 = SALayer()
        self.sa4 = SALayer()

        # self.selfatten=SelfAttention(512+256)

        self.resnet = ResNet18()

    def forward(self, x):
        y, uv = tensor2y_uv(x)

        res_64, res_128, res_256, res_512 = self.resnet(uv)

        rdb_256 = self.rdb_conv_1(y)
        rdb_256 = self.BN_1(rdb_256)
        rdb_256 = self.rdb_1(rdb_256)

        rdb_128 = self.rdb_conv_2(rdb_256)
        rdb_128 = self.BN_2(rdb_128)
        rdb_128 = self.rdb_2(rdb_128)

        rdb_64 = self.rdb_conv_3(rdb_128)
        rdb_64 = self.BN_3(rdb_64)
        rdb_64 = self.rdb_3(rdb_64)

        temp = torch.cat((rdb_64, res_64), dim=1)
        temp = self.se1(temp)
        temp = self.sa1(temp) * temp
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.upsample_2(de_feature_1_conv)

        temp = torch.cat((de_feature_2, rdb_128, res_128), dim=1)
        temp = self.se2(temp)
        temp = self.sa2(temp) * temp

        de_feature_2_conv = self.right_conv_2(temp)
        de_feature_3 = self.upsample_3(de_feature_2_conv)

        temp = torch.cat((de_feature_3, rdb_256, res_256), dim=1)
        temp = self.se3(temp)
        temp = self.sa3(temp) * temp

        de_feature_3_conv = self.right_conv_3(temp)
        de_feature_4 = self.upsample_4(de_feature_3_conv)

        temp = torch.cat((de_feature_4, res_512), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)

        out = self.right_conv_5(de_feature_4_conv)
        out = self.tanh(out)
        return out


if __name__ == "__main__":

    def test():
        net = YUV_Net()
        y = net(torch.randn(1, 3, 512, 512))

    test()
