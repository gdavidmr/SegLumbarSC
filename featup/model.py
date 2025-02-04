import torchvision.transforms as T
from PIL import Image

# from featup.util import norm, unnorm
# from featup.plotting import plot_feats, plot_lang_heatmaps

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationCNN(nn.Module):
    def __init__(self, input_channels=384, output_channels=1):
        super(SegmentationCNN, self).__init__()

        # Encoder
        self.enc1 = nn.Conv3d(input_channels, 256, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.reduce3 = nn.Conv3d(128, 64, kernel_size=1, stride=1)  # Reduce concatenated channels
        self.dec3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upconv2 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.reduce2 = nn.Conv3d(192, 64, kernel_size=1, stride=1)  # Reduce concatenated channels
        self.dec2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upconv1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.reduce1 = nn.Conv3d(320, 64, kernel_size=1, stride=1)  # Reduce concatenated channels
        self.dec1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)

        # Output
        self.out = nn.Conv3d(64, output_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(self.pool(x1)))
        x3 = F.relu(self.enc3(self.pool(x2)))

        # Bottleneck
        x_bottleneck = F.relu(self.bottleneck(self.pool(x3)))

        # Decoder
        x_up3 = F.relu(self.upconv3(x_bottleneck))
        # print(x_up3.shape, x3.shape)
        if x_up3.shape[-1] < x3.shape[-1]:
            pad_size = x3.shape[-1] - x_up3.shape[-1]
            x_up3 = F.pad(x_up3, (0, pad_size))
        else:
            pad_size = x_up3.shape[-1] - x3.shape[-1]
            x3 = F.pad(x3, (0, pad_size))
        x_up3 = torch.cat((x_up3, x3), dim=1)  # Concatenate along channel dimension
        x_up3 = F.relu(self.reduce3(x_up3))  # Reduce concatenated channels
        x_up3 = F.relu(self.dec3(x_up3))
        x_up2 = F.relu(self.upconv2(x_up3))
        # print(x_up2.shape, x2.shape)
        x_up2 = torch.cat((x_up2, x2), dim=1)  # Concatenate along channel dimension
        x_up2 = F.relu(self.reduce2(x_up2))  # Reduce concatenated channels
        x_up2 = F.relu(self.dec2(x_up2))

        x_up1 = F.relu(self.upconv1(x_up2))
        x_up1 = torch.cat((x_up1, x1), dim=1)  # Concatenate along channel dimension
        x_up1 = F.relu(self.reduce1(x_up1))  # Reduce concatenated channels
        x_up1 = F.relu(self.dec1(x_up1))

        # Output layer
        output = self.out(x_up1)
        return output


# Example usage
if __name__ == "__main__":
    model = SegmentationCNN(input_channels=384, output_channels=1).cuda()
    x = torch.randn(1, 384, 224, 224, 20).cuda()  # Example input tensor
    output = model(x)
    print("Output shape:", output.shape)