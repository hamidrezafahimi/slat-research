import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    """
    Encoder-decoder network for unsupervised segmentation.
    Expects two inputs (image and pattern), concatenated along the channel axis.
    Produces a single-channel mask.
    """
    def __init__(self, in_channels=2):
        """
        in_channels=2 means:
          1 channel for the input image,
          1 channel for the pattern image.
        """
        super(EncoderDecoder, self).__init__()
        
        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Bottleneck
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, 
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, 
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, 
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # final mask in [0,1]
        )
        
    def forward(self, x, p):
        """
        Forward pass.
          x: input image, shape (B, 1, H, W)
          p: pattern image, shape (B, 1, H, W)
        Returns:
          mask: shape (B, 1, H, W)
        """
        # Concatenate along channel dimension => shape (B, 2, H, W)
        inp = torch.cat((x, p), dim=1)
        features = self.encoder(inp)
        mask = self.decoder(features)
        return mask

def segmentation_loss(x, p, m, alpha=0.01, beta=0.1):
    """
    x     -> input image (B,1,H,W)
    p     -> pattern image (B,1,H,W)
    m     -> predicted mask (B,1,H,W)
    alpha -> weight for binary penalty
    beta  -> weight for coverage penalty
    """
    # 1) Pixel-wise similarity loss: encourages M to be high where X matches P
    diff = (x - p)**2
    L_match = (m * diff).mean()  # Only penalize where M is high

    # 2) Binary constraint: penalizes non-binary values
    L_bin = (m * (1 - m)).mean()

    # 3) Coverage loss: prevents full black masks
    L_cov = (0.5 - m.mean()).abs()  # Encourages ~50% of pixels to be kept

    return L_match + alpha * L_bin + beta * L_cov
