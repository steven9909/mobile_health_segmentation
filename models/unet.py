import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Initialize UNet

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
        """
        super().__init__()
        self.down_blocks = nn.ModuleList(
            [
                UNetBlock(in_channels, 64),
                UNetBlock(64, 128),
                UNetBlock(128, 256),
                UNetBlock(256, 512),
            ]
        )

        self.bottleneck = UNetBlock(512, 1024)

        self.up_blocks = nn.ModuleList(
            [
                nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ]
        )

        self.blocks = nn.ModuleList(
            [
                UNetBlock(1024, 512),
                UNetBlock(512, 256),
                UNetBlock(256, 128),
                UNetBlock(128, 64),
            ]
        )

        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        """Perform forward pass on UNet
        If input is of shape (B, C, H, W), the output is also of shape (B, C, H, W)

        Args:
            x (Tensor): input

        Returns:
            Tensor: output of the same shape as input
        """
        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)
            x = self.max_pool(x)

        x = self.bottleneck(x)

        for i, (up_block, block) in enumerate(zip(self.up_blocks, self.blocks)):
            # Concatenate on the channel dimension
            x = torch.cat((skip_connections[-(i + 1)], up_block(x)), dim=1)
            x = block(x)

        return self.head(x)


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Initialize UNetBlock - double conv blocks with ReLU

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


if __name__ == "__main__":
    unet = UNet(3, 1)
    input = torch.randn((1, 3, 256, 256))

    assert unet(input).shape == (1, 1, 256, 256)
