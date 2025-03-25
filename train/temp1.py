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

def smooth_threshold_loss(image, template_mask, threshold_pattern, k=10.0, lambda_tv=1.0, lambda_seg=1.0):
    """
    image: the original depth image tensor [B, 1, H, W]
    template_mask: the target template mask tensor [B, 1, H, W]
    threshold_pattern: the network's output threshold pattern [B, 1, H, W]
    k: controls the steepness of the sigmoid used for soft thresholding.
    lambda_tv: weight for the Total Variation (TV) loss; increased to strongly enforce smoothness.
    lambda_seg: weight for the segmentation loss (comparing the masked image to the template mask).
    
    Returns:
        A combined loss that heavily penalizes high gradients in the threshold pattern.
    """
    # Compute a differentiable soft mask.
    # When image < threshold_pattern, (image - threshold_pattern) is negative, 
    # so sigmoid(k*(...)) is near 0 and 1 - sigmoid is near 1.
    # When image > threshold_pattern, the mask goes to 0.
    mask = 1 - torch.sigmoid(k * (image - threshold_pattern))
    masked_image = image * mask

    # Segmentation loss: encourage some alignment with the template mask.
    seg_loss = F.l1_loss(masked_image, template_mask)

    # Total Variation (TV) loss: penalizes large differences between neighboring pixels.
    tv_h = torch.mean(torch.abs(threshold_pattern[:, :, :, :-1] - threshold_pattern[:, :, :, 1:]))
    tv_v = torch.mean(torch.abs(threshold_pattern[:, :, :-1, :] - threshold_pattern[:, :, 1:, :]))
    tv_loss = tv_h + tv_v

    # Combined loss: here the TV term dominates.
    return lambda_seg * seg_loss + lambda_tv * tv_loss

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
            loss_value = segmentation_loss(x_batch, template_mask, th_pattern)
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

# -------------------------
# Run Script Example
# -------------------------
if __name__ == '__main__':
    # Update these paths as needed
    depth_images_dir = '/path/to/depth_images'
    template_masks_dir = '/path/to/template_masks'
    checkpoints_dir = './checkpoints'
    
    # Optionally add normalization or other transforms.
    transform = transforms.Compose([
        transforms.ToTensor(),
        # e.g., transforms.Normalize(mean, std)
    ])
    
    # Create model and dataloader for training
    model, dataloader = create_model(depth_images_dir, template_masks_dir, batch_size=4, transform=transform)
    
    # Train the model (update hyperparameters as needed)
    train_model(model, dataloader, epochs=20, lr=1e-4, device='cuda', checkpoint_interval=5, checkpoint_dir=checkpoints_dir)
    
    # Evaluate the model on a single image (update file paths)
    model_checkpoint = os.path.join(checkpoints_dir, "checkpoint_epoch_20.pth")
    input_image_file = '/path/to/depth_images/sample.jpg'
    template_mask_file = '/path/to/template_masks/sample.jpg'
    output_directory = './evaluation_output'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate_model(model_checkpoint, input_image_file, template_mask_file, output_directory, device=device)
