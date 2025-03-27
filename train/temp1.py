import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import LSQBivariateSpline

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

# -----------------------------
# Your Provided Spline Fitter (unchanged)
# -----------------------------
class VisualSplineFit:
    def __init__(self, x_coeffs=4, degree_x=3, degree_y=3):
        """
        A simple 2D spline-fitting helper for a grayscale image of shape (H, W).
        """
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.num_coeffs_x = x_coeffs
        self.params_init = False

    def initParams(self, img_shape):
        self.H, self.W = img_shape  # shape of the image
        self.num_coeffs_y = int(np.round((float(self.H) / self.W) * self.num_coeffs_x))

        X, Y = np.meshgrid(
            np.linspace(0, 1, self.W),
            np.linspace(0, 1, self.H)
        )
        self.x = X.ravel()
        self.y = Y.ravel()

        self.num_knots_x = self.num_coeffs_x + self.degree_x + 1
        self.num_knots_y = self.num_coeffs_y + self.degree_y + 1
        self.num_inner_knots_x = self.num_knots_x - 2 * self.degree_x
        self.num_inner_knots_y = self.num_knots_y - 2 * self.degree_y

        self.inner_knots_x = np.linspace(0, 1, self.num_inner_knots_x + 2)[1:-1]
        self.inner_knots_y = np.linspace(0, 1, self.num_inner_knots_y + 2)[1:-1]
        self.params_init = True

    def fit(self, img):
        """
        Fits a 2D B-spline to the image data (H x W) using LSQBivariateSpline.
        Returns the fitted surface as a NumPy array (H x W).
        """
        if not self.params_init:
            self.initParams(img.shape)

        z = img.astype(np.float64).ravel()
        spline = LSQBivariateSpline(
            self.x,
            self.y,
            z,
            tx=self.inner_knots_x,
            ty=self.inner_knots_y,
            kx=self.degree_x,
            ky=self.degree_y
        )
        x_sorted = np.linspace(0, 1, self.W)
        y_sorted = np.linspace(0, 1, self.H)
        fitted_2d = spline(x_sorted, y_sorted)
        if fitted_2d.shape[1] == self.H and fitted_2d.shape[0] == self.W:
            fitted_2d = fitted_2d.T
        return fitted_2d

# -----------------------------
# Custom Autograd Function: SplineFitSTE
# -----------------------------
class SplineFitSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output_img):
        """
        Forward pass: Convert the network output to NumPy, call VisualSplineFit.fit,
        then convert the fitted result back to a torch tensor.
        """
        # output_img: [B, 1, H, W]
        output_np = output_img.detach().cpu().numpy()
        B, C, H, W = output_np.shape
        vsf = VisualSplineFit(x_coeffs=4, degree_x=3, degree_y=3)
        fitted_list = []
        for i in range(B):
            sample = output_np[i, 0, :, :]
            fitted = vsf.fit(sample)
            fitted_list.append(fitted)
        fitted_array = np.stack(fitted_list, axis=0)  # shape: [B, H, W]
        fitted_tensor = torch.from_numpy(fitted_array).unsqueeze(1).to(output_img.device).type_as(output_img)
        return fitted_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass the gradient through unchanged.
        return grad_output

# -----------------------------
# Updated LossCalculator using SplineFitSTE
# -----------------------------
class LossCalculator(nn.Module):
    """
    This loss calculator:
      - Passes the network's raw output (threshold pattern) through SplineFitSTE to obtain a smooth spline.
      - Uses the fitted spline to generate a soft mask via a sigmoid.
      - Computes the loss as 1 minus the overlap between the masked image and the template mask.
    """
    def __init__(self, k=10.0):
        super(LossCalculator, self).__init__()
        self.k = k

    def forward(self, input_image, tmask, output_img):
        # Use the custom autograd function for spline fitting.
        fitted_spline = SplineFitSTE.apply(output_img)
        # Generate a soft mask: pixels where input_image is below the fitted spline are preserved.
        soft_mask = 1 - torch.sigmoid(self.k * (input_image - fitted_spline))
        masked_image = input_image * soft_mask
        # Compute overlap: sum over template area divided by template area.
        overlap_ratio = torch.sum(tmask * masked_image) / (torch.sum(tmask) + 1e-8)
        loss = 1 - overlap_ratio
        return loss

# -----------------------------
# Revised train_model using LossCalculator
# -----------------------------
def train_model(model, dataloader, epochs=10, lr=1e-4, device='cuda', 
                checkpoint_interval=5, checkpoint_dir='./checkpoints'):
    """
    Trains the model using the LossCalculator (which leverages the spline fitter with a straight-through estimator).
    
    Parameters:
      model: network model (expects input: [B,1,H,W])
      dataloader: yields (input_image, template_mask) pairs (both [B,1,H,W])
      epochs: number of training epochs
      lr: learning rate
      device: device to run on ('cuda' or 'cpu')
      checkpoint_interval: number of epochs between checkpoints
      checkpoint_dir: directory to save checkpoints
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    loss_calc = LossCalculator(k=10.0)
    loss_calc.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch_idx, (x_batch, template_mask) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            template_mask = template_mask.to(device)
            
            optimizer.zero_grad()
            # Forward pass: get the raw threshold pattern from the model.
            output = model(x_batch)
            # Compute loss using LossCalculator.
            loss_value = loss_calc(x_batch, template_mask, output)
            
            loss_value.backward()
            optimizer.step()
            
            epoch_loss += loss_value.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
