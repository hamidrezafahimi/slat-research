import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# -------------------------
# UNet Model Definition
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

# -------------------------
# Dataset Class
# -------------------------
class ImagePatternDataset(Dataset):
    """
    Dataset for loading depth images and template masks.
    
    Parameters:
      depth_img_paths (list): List of file paths to depth images.
      template_mask_paths (list): List of file paths to template masks.
      transform (callable, optional): Optional transform to apply to images.
    """
    def __init__(self, depth_img_paths, template_mask_paths, transform=None):
        if len(depth_img_paths) != len(template_mask_paths):
            raise ValueError("The number of depth images and template masks must match.")
        self.depth_img_paths = depth_img_paths
        self.template_mask_paths = template_mask_paths
        self.transform = transform if transform is not None else transforms.ToTensor()
        
    def __len__(self):
        return len(self.depth_img_paths)
    
    def __getitem__(self, idx):
        depth_img = Image.open(self.depth_img_paths[idx]).convert("L")
        template_mask = Image.open(self.template_mask_paths[idx]).convert("L")
        
        depth_img = self.transform(depth_img)
        template_mask = self.transform(template_mask)
        
        return depth_img, template_mask

import torch
import torch.nn.functional as F

def differentiable_entropy(x, num_bins=64, sigma=0.02):
    """
    Compute a differentiable approximation to Shannon entropy.
    
    Args:
      x: tensor of shape [B,1,H,W] with values in [0,1].
      num_bins: number of histogram bins.
      sigma: controls the softness of the histogram binning.
      
    Returns:
      Average normalized entropy over the batch (scalar tensor).
    """
    B, C, H, W = x.shape
    N = H * W
    x_flat = x.view(B, N)  # shape: [B, N]
    # Create bin centers between 0 and 1.
    bin_centers = torch.linspace(0, 1, num_bins, device=x.device, dtype=x.dtype).view(1, num_bins)
    # Compute soft assignments: for each pixel, weight for each bin.
    x_expanded = x_flat.unsqueeze(2)  # shape: [B, N, 1]
    bin_centers_expanded = bin_centers.unsqueeze(1)  # shape: [B, 1, num_bins]
    weights = torch.exp(-((x_expanded - bin_centers_expanded)**2) / (2 * sigma**2))  # [B, N, num_bins]
    # Sum over pixels to get a soft histogram.
    hist = weights.sum(dim=1)  # [B, num_bins]
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
    # Compute entropy: H = - sum(p * log(p)).
    entropy = - (hist * torch.log(hist + 1e-8)).sum(dim=1)  # [B]
    # Normalize entropy by log(num_bins) so that maximum entropy is 1.
    normalized_entropy = entropy / torch.log(torch.tensor(num_bins, device=x.device, dtype=x.dtype))
    return normalized_entropy.mean()

def compute_smoothness_loss(threshold_pattern, max_tv=10000, max_lap_var=0.02, num_bins=64, sigma=0.02):
    """
    Compute a smoothness loss from several differentiable metrics on the threshold pattern.
    
    Args:
      threshold_pattern: [B,1,H,W] network output (assumed to be in [0,1]).
      max_tv: maximum total variation (for normalization).
      max_lap_var: maximum Laplacian variance (for normalization).
      num_bins: number of bins for differentiable entropy.
      sigma: bandwidth for the soft histogram.
      
    Returns:
      A scalar tensor representing the smoothness loss (higher means more edge-rich/complex).
    """
    # Normalize threshold_pattern to [0,1] (should be nearly so, but for stability)
    tp_min = threshold_pattern.amin(dim=[1,2,3], keepdim=True)
    tp_max = threshold_pattern.amax(dim=[1,2,3], keepdim=True)
    norm_tp = (threshold_pattern - tp_min) / (tp_max - tp_min + 1e-8)
    
    # 1. Sobel Edge Score.
    sobel_kernel_x = torch.tensor([[1, 0, -1],
                                   [2, 0, -2],
                                   [1, 0, -1]], dtype=norm_tp.dtype, device=norm_tp.device).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]], dtype=norm_tp.dtype, device=norm_tp.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(norm_tp, sobel_kernel_x, padding=1)
    grad_y = F.conv2d(norm_tp, sobel_kernel_y, padding=1)
    edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    edge_mean = edge_magnitude.mean(dim=[1,2,3])
    # Normalize: maximum edge strength approximated by 1.41 (L2 norm of [1,2,1]).
    edge_score = torch.clamp(edge_mean / 1.41, 0, 1)
    
    # 2. Laplacian Variance Score.
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=norm_tp.dtype, device=norm_tp.device).view(1, 1, 3, 3)
    lap = F.conv2d(norm_tp, laplacian_kernel, padding=1)
    lap_var = lap.var(dim=[1,2,3])
    lap_score = torch.clamp(torch.log1p(lap_var) / torch.log1p(torch.tensor(max_lap_var, device=norm_tp.device, dtype=norm_tp.dtype)), 0, 1)
    
    # 3. Entropy Score.
    ent = differentiable_entropy(norm_tp, num_bins=num_bins, sigma=sigma)
    entropy_score = torch.clamp(ent / 8.0, 0, 1)
    
    # 4. Total Variation Score.
    tv_h = torch.abs(norm_tp[:, :, 1:, :] - norm_tp[:, :, :-1, :]).sum(dim=[1,2,3])
    tv_v = torch.abs(norm_tp[:, :, :, 1:] - norm_tp[:, :, :, :-1]).sum(dim=[1,2,3])
    tv = tv_h + tv_v
    tv_score = torch.clamp(torch.log1p(tv) / torch.log1p(torch.tensor(max_tv, device=norm_tp.device, dtype=norm_tp.dtype)), 0, 1)
    
    combined_loss = (edge_score + lap_score + entropy_score + tv_score) / 4.0
    return combined_loss.mean()

def final_loss(image, template_mask, threshold_pattern, 
               k=10.0, lambda_seg=1.0, lambda_smooth=1.0,
               max_tv=10000, max_lap_var=0.02, num_bins=64, sigma=0.02):
    """
    Final loss function for training.
    
    Args:
      image: [B,1,H,W] original depth image.
      template_mask: [B,1,H,W] template mask.
      threshold_pattern: [B,1,H,W] network output (should be in [0,1]).
      k: steepness factor for soft thresholding.
      lambda_seg: weight for the segmentation loss (soft-threshold masked image vs. template mask).
      lambda_smooth: weight for the smoothness loss (edge, Laplacian, entropy, TV).
      lambda_bin: weight for the binary penalty (prevent threshold values from nearing 0 or 1).
      margin: threshold margin below which (or above 1-margin) values are penalized.
      max_tv, max_lap_var, num_bins, sigma: parameters for smoothness loss normalization.
      
    Returns:
      A scalar tensor: the final loss.
    """
    # Compute a differentiable soft mask.
    mask = 1 - torch.sigmoid(k * (image - threshold_pattern))
    masked_image = image * mask
    seg_loss = F.l1_loss(masked_image, template_mask)
    
    # Compute the smoothness loss on the threshold pattern.
    smooth_loss = compute_smoothness_loss(threshold_pattern, max_tv=max_tv, max_lap_var=max_lap_var, num_bins=num_bins, sigma=sigma)

    total_loss = lambda_seg * seg_loss + lambda_smooth * smooth_loss
    return total_loss

# -------------------------
# Training Function
# -------------------------
def train_model(model, dataloader, epochs=10, lr=1e-4, device='cuda', checkpoint_interval=5, checkpoint_dir='./checkpoints'):
    """
    Trains the UNet model with the differentiable segmentation loss.
    
    Parameters:
      model             -> instance of UNet.
      dataloader        -> yields (depth_image_batch, template_mask_batch).
      epochs            -> number of training epochs.
      lr                -> learning rate.
      device            -> 'cuda' or 'cpu'.
      checkpoint_interval -> save model every N epochs.
      checkpoint_dir    -> directory to save checkpoint files.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch_idx, (x_batch, template_mask) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            template_mask = template_mask.to(device)

            optimizer.zero_grad()
            # Forward pass: produce threshold pattern from depth image.
            th_pattern = model(x_batch)
            loss_value = final_loss(x_batch, template_mask, th_pattern)
            loss_value.backward()
            optimizer.step()

            epoch_loss += loss_value.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Save checkpoint if needed
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

# -------------------------
# Model Creation Function
# -------------------------
def create_model(depth_img_directory, template_mask_directory, batch_size=4, transform=None):
    """
    Creates a UNet model and DataLoader for the paired dataset.
    
    Parameters:
      depth_img_directory     -> directory containing depth images (.jpg).
      template_mask_directory -> directory containing template masks (.jpg).
      
    Returns:
      model: an instance of UNet.
      dataloader: DataLoader for the paired dataset.
    """
    depth_img_paths = [os.path.join(depth_img_directory, f) for f in os.listdir(depth_img_directory) if f.endswith(".jpg")]
    template_mask_paths = [os.path.join(template_mask_directory, f) for f in os.listdir(template_mask_directory) if f.endswith(".jpg")]

    depth_img_paths.sort()
    template_mask_paths.sort()

    if len(depth_img_paths) != len(template_mask_paths):
        raise ValueError(f"Number of depth images ({len(depth_img_paths)}) and template masks ({len(template_mask_paths)}) do not match!")
    
    dataset = ImagePatternDataset(depth_img_paths, template_mask_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = UNet(n_channels=1, n_classes=1)
    return model, dataloader

# -------------------------
# Evaluation Script
# -------------------------
def evaluate_model(model_checkpoint_path, input_image_path, template_mask_path, output_dir, device='cuda'):
    """
    Loads a trained UNet model and evaluates it on a single input image.
    
    Saves:
      - The masked (thresholded) image using the soft-thresholding.
      - The threshold pattern.
      - The template mask.
    """
    model = UNet(n_channels=1, n_classes=1)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    input_image = Image.open(input_image_path).convert("L")
    input_tensor = transform(input_image).unsqueeze(0).to(device)  # shape: [1, 1, H, W]
    
    with torch.no_grad():
        threshold_pattern = model(input_tensor)
    
    # Use soft thresholding for evaluation as well.
    k = 10.0
    mask = 1 - torch.sigmoid(k * (input_tensor - threshold_pattern))
    masked_tensor = input_tensor * mask
    
    # Load the template mask
    template_mask = Image.open(template_mask_path).convert("L")
    
    to_pil = transforms.ToPILImage()
    masked_image_pil = to_pil(masked_tensor.squeeze(0).cpu())
    threshold_pattern_pil = to_pil(threshold_pattern.squeeze(0).cpu())
    
    os.makedirs(output_dir, exist_ok=True)
    
    masked_image_path = os.path.join(output_dir, "masked_image.png")
    threshold_pattern_path = os.path.join(output_dir, "threshold_pattern.png")
    template_mask_path_out = os.path.join(output_dir, "template_mask.png")
    
    masked_image_pil.save(masked_image_path)
    threshold_pattern_pil.save(threshold_pattern_path)
    template_mask.save(template_mask_path_out)
    
    print(f"Saved masked image to {masked_image_path}")
    print(f"Saved threshold pattern to {threshold_pattern_path}")
    print(f"Saved template mask to {template_mask_path_out}")

