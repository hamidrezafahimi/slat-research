import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Two consecutive convolution layers with BatchNorm and ReLU."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # optional normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double convolution."""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double convolution. Uses bilinear upsampling by default."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # Learnable upsampling if bilinear is not preferred.
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust size (if needed) to match skip connection from encoder
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution to get desired number of output channels."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

# -------------------------
# U-Net Architecture
# -------------------------

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        """
        n_channels: number of input channels (1 for monochrome/depth image)
        n_classes: number of output channels (1 for threshold pattern)
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 1024, H/16, W/16]
        # Decoder path with skip connections
        x = self.up1(x5, x4)  # [B, 512, H/8, W/8]
        x = self.up2(x, x3)   # [B, 256, H/4, W/4]
        x = self.up3(x, x2)   # [B, 128, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64, H, W]
        x = self.outc(x)      # [B, n_classes, H, W]
        # Use sigmoid to constrain threshold values between 0 and 1.
        return torch.sigmoid(x)


import torch.nn.functional as F


# ================================
# Loss Function
# ================================
# def segmentation_loss(input_image, template_mask, threshold_pattern, alpha=0.01):
#     """
#     Compute segmentation loss.
    
#     - Create a masked image using the threshold pattern.
#     - Use L1Loss between the masked image and template mask.
#     - Add a binary penalty (alpha weighted) on the threshold pattern.
#     """
#     # Create masked image: pixels where input < threshold keep their value; else zero.
#     masked_image = torch.where(input_image < threshold_pattern, input_image, torch.tensor(0.0, device=input_image.device))
#     criterion = nn.L1Loss()
#     base_loss = criterion(masked_image, template_mask)
#     # Binary penalty to encourage threshold values to be near 0 or 1
#     penalty = alpha * torch.mean(threshold_pattern * (1 - threshold_pattern))
#     return base_loss + penalty


import numpy as np
def segmentation_loss(image, template_mask, threshold_pattern):
    HUGE_METRIC = 1e9      # Value to assign if skip condition is met.
    binary_output = (image > threshold_pattern).astype(np.uint8) * 255
    # Check trivial condition: if binary_output is completely 0 or 255.
    if np.all(binary_output == 0) or np.all(binary_output == 255):
        return HUGE_METRIC, None

    # Create masked image: keep original pixel where binary_output is 0.
    masked_img = np.where(binary_output == 0, image, 0).astype(np.uint8)
    total_pixels = masked_img.size
    # # Unmasked pixels: pixels with value > 5.
    mask = (masked_img > 4)
    unmasked_count = np.count_nonzero(mask)
    unmasked_fraction = unmasked_count / total_pixels

    # Compute overlap metric with the template mask.
    overlap_ratio = np.sum(masked_img == template_mask) / total_pixels
    overlap_metric = 1 - overlap_ratio

    # Skip condition if too few unmasked pixels or unmasked fraction < 10%.
    if unmasked_count < 6 or unmasked_fraction < 0.1:
        # print(f"Skip condition -- too few unmasked pixels ({unmasked_count}) or unmasked fraction ({unmasked_fraction}) < 10%")
        return HUGE_METRIC * overlap_metric, None

    # New skip condition: if overlap_ratio is less than 20% then skip then skip.
    if overlap_ratio < 0.2:
        # print(f"Skip condition -- overlap_ratio ({overlap_ratio}) is less than 20%")
        return HUGE_METRIC * overlap_metric, None
    
    return overlap_metric, masked_img