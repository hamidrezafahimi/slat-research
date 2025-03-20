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


import torch.nn.functional as F

def segmentation_loss(x, p, m, alpha=0.01, beta=0.1, gamma=1.0):
    """
    x     -> input depth image (B,1,H,W)
    p     -> pattern depth image (B,1,H,W)
    m     -> predicted mask (B,1,H,W) with:
               0 (black) for foreground (objects)
               1 (white) for background (ground)
    alpha -> weight for binary penalty
    beta  -> weight for coverage penalty
    gamma -> weight for edge penalty
    
    The loss has four parts:
      1) Gradient similarity loss: Encourages m to be high (background) where
         the gradients of x and p are similar.
      2) Edge penalty: Penalizes high m values in regions of x that exhibit strong edges.
      3) Binary constraint: Encourages the mask to be near-binary.
      4) Coverage loss: Prevents trivial solutions by encouraging ~50% of pixels to be background.
    """
    
    # --- Compute gradients using finite differences ---
    # Horizontal gradients (difference along width)
    grad_x_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    grad_p_h = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1])
    
    # Vertical gradients (difference along height)
    grad_x_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    grad_p_v = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])
    
    # Pad the gradients to restore original shape (B,1,H,W)
    grad_x_h = F.pad(grad_x_h, (0, 1, 0, 0), mode='replicate')
    grad_p_h = F.pad(grad_p_h, (0, 1, 0, 0), mode='replicate')
    grad_x_v = F.pad(grad_x_v, (0, 0, 0, 1), mode='replicate')
    grad_p_v = F.pad(grad_p_v, (0, 0, 0, 1), mode='replicate')
    
    # Total gradient (simple sum of horizontal and vertical components)
    grad_x_total = grad_x_h + grad_x_v
    grad_p_total = grad_p_h + grad_p_v
    
    # --- Loss Components ---
    # 1) Gradient similarity loss:
    #   In background areas (m high), x should have a similar gradient to p.
    L_grad_sim = (m * (grad_x_total - grad_p_total) ** 2).mean()
    
    # 2) Edge penalty:
    #   In regions with strong gradients in x (i.e. edges), the mask should be low (foreground).
    #   If m is high in these regions, we add a penalty.
    L_edge = (m * (grad_x_total ** 2)).mean()
    
    # 3) Binary constraint: encourages m to be close to 0 or 1.
    L_bin = (m * (1 - m)).mean()
    
    # 4) Coverage loss: encourages the overall background area to be around 50% of the image.
    L_cov = (0.5 - m.mean()).abs()
    
    return L_grad_sim + gamma * L_edge + alpha * L_bin + beta * L_cov
