import torch
import torch.nn as nn
import torch.optim as optim

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

def segmentation_loss(x, p, m, alpha=0.01):
    """
    x     -> input image (B,1,H,W)
    p     -> pattern image (B,1,H,W)
    m     -> predicted mask (B,1,H,W)
    alpha -> weight for binary penalty
    """
    # 1) Mean squared difference weighted by the mask M
    #    If M ~ 1, we want X ~ P to avoid high penalty. If X != P, the cost is high.
    diff = m * (x - p)**2
    mse_term = diff.mean()

    # 2) Encourage M to be near 0 or 1
    #    M(1-M) is maximized at M=0.5. Minimizing it pushes M to 0 or 1.
    bin_penalty = m * (1 - m)
    bin_term = bin_penalty.mean()

    return mse_term + alpha * bin_term

def train_model(model, dataloader, epochs=10, alpha=0.01, lr=1e-4, device='cuda'):
    """
    model     -> instance of EncoderDecoder
    dataloader-> yields (image_batch, pattern_batch)
    epochs    -> number of training epochs
    alpha     -> hyperparameter for binary penalty
    lr        -> learning rate
    device    -> 'cuda' or 'cpu'
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch_idx, (x_batch, p_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)

            optimizer.zero_grad()
            m_batch = model(x_batch, p_batch)   # forward pass
            loss_value = segmentation_loss(x_batch, p_batch, m_batch, alpha=alpha)
            loss_value.backward()               # backprop
            optimizer.step()                    # update parameters

            epoch_loss += loss_value.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
