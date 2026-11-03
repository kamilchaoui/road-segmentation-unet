import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):
    """
    (Conv -> BN -> ReLU) * 2
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """
    Up-sampling block:
    upsample (ConvTranspose2d) -> concat with skip -> DoubleConv
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle possible off-by-one due to odd sizes: center-crop skip
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            skip = skip[
                :,
                :,
                diff_y // 2 : diff_y // 2 + x.size(2),
                diff_x // 2 : diff_x // 2 + x.size(3),
            ]

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net with ResNet34 encoder (pretrained on ImageNet).

    Input:  B x in_channels x H x W
    Output: B x out_channels x H x W (logits, no sigmoid)

    Usage:
        model = UNet(in_channels=3, out_channels=1)

    Notes:
        - in_channels must be 3 in this version (RGB).
        - base_c is kept for API compatibility but not used directly.
    """

    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super(UNet, self).__init__()

        if in_channels != 3:
            raise ValueError("This UNet with ResNet34 encoder currently supports in_channels=3 only.")

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)

        # Reduce downsampling: conv1 stride=1, remove maxpool
        backbone.conv1.stride = (1, 1)
        self.encoder_conv1 = backbone.conv1
        self.encoder_bn1 = backbone.bn1
        self.encoder_relu = backbone.relu
        self.encoder_maxpool = nn.Identity()  # no /2 here

        self.encoder_layer1 = backbone.layer1  # 64 channels, H x W
        self.encoder_layer2 = backbone.layer2  # 128 channels, H/2 x W/2
        self.encoder_layer3 = backbone.layer3  # 256 channels, H/4 x W/4
        self.encoder_layer4 = backbone.layer4  # 512 channels, H/8 x W/8

        # -------- Decoder --------
        # We decode from /8 -> /4 -> /2 -> /1 (3 upsamples)
        self.up3 = UpBlock(in_channels=512, skip_channels=256, out_channels=256)  # /8 -> /4
        self.up2 = UpBlock(in_channels=256, skip_channels=128, out_channels=128)  # /4 -> /2
        self.up1 = UpBlock(in_channels=128, skip_channels=64, out_channels=64)  # /2 -> /1

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        # x: B x 3 x H x W

        x0 = self.encoder_conv1(x)  # B x 64 x H x W
        x0 = self.encoder_bn1(x0)
        x0 = self.encoder_relu(x0)

        x1 = self.encoder_maxpool(x0)  # identity -> still B x 64 x H x W
        x1 = self.encoder_layer1(x1)  # B x 64  x H   x W
        x2 = self.encoder_layer2(x1)  # B x 128 x H/2 x W/2
        x3 = self.encoder_layer3(x2)  # B x 256 x H/4 x W/4
        x4 = self.encoder_layer4(x3)  # B x 512 x H/8 x W/8  (bottleneck)

        # -------- Decoder --------
        d3 = self.up3(x4, x3)  # B x 256 x H/4 x W/4
        d2 = self.up2(d3, x2)  # B x 128 x H/2 x W/2
        d1 = self.up1(d2, x1)  # B x 64  x H   x W

        logits = self.final_conv(d1)  # B x out_channels x H x W
        return logits
