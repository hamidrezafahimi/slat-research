import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spline_fit import VisualSplineFit
import cv2


# -----------------------------
# Loss Calculator Using the Spline Fitter
# -----------------------------
class LossCalculator(nn.Module):
    """
    LossCalculator passes the network's output (threshold pattern) through a spline fitter
    to obtain a smooth 2D curve, uses that fitted spline to generate a soft mask for the input image,
    and computes the loss based on the overlap between the template masked image and the masked image.
    """
    def __init__(self, k=10.0, x_coeffs=4, degree_x=3, degree_y=3):
        super(LossCalculator, self).__init__()
        self.k = k
        self.splineFitter = VisualSplineFit(x_coeffs=x_coeffs, degree_x=degree_x, degree_y=degree_y)

    def forward(self, input_image, tmask, output_img):
        """
        Args:
          input_image: [B, 1, H, W] tensor (normalized depth image).
          tmask: [B, 1, H, W] tensor (template masked image as a binary mask, values 0 or 1).
          output_img: [B, 1, H, W] tensor (raw threshold pattern output by the network).
        Returns:
          A scalar loss value (1 - overlap ratio).
        """
        B, C, H, W = output_img.shape
        fitted_splines = []
        # Convert the network output to NumPy for spline fitting.
        output_img_np = output_img.detach().cpu().numpy()  # shape: [B, 1, H, W]
        for i in range(B):
            # Extract the i-th sample as a 2D image.
            sample_out = output_img_np[i, 0, :, :]
            fitted = self.splineFitter.fit(sample_out)
            fitted_splines.append(fitted)
        # Stack fitted splines into a NumPy array of shape [B, H, W].
        fitted_splines = np.stack(fitted_splines, axis=0)
        # Convert fitted splines back to a torch tensor with shape [B, 1, H, W].
        fitted_spline_tensor = torch.from_numpy(fitted_splines).unsqueeze(1).to(output_img.device).type_as(output_img)
        
        # Generate a differentiable soft mask.
        # For each pixel: if input_image < fitted_spline then the sigmoid is near 1; otherwise near 0.
        soft_mask = 1 - torch.sigmoid(self.k * (input_image - fitted_spline_tensor))
        masked_image = input_image * soft_mask
        
        # Compute the overlap ratio between the template mask and the masked image.
        # Here we sum the masked image values over the template area and normalize by the total "on" area.
        overlap_ratio = torch.sum(tmask * masked_image) / (torch.sum(tmask) + 1e-8)
        
        # Our loss is 1 - overlap_ratio (we want high overlap, i.e. loss to be low).
        loss = 1 - overlap_ratio
        return loss

if __name__ == "__main__":
    # Read images in grayscale (uint8)
    input_image_cv = cv2.imread("00000001.jpg", cv2.IMREAD_GRAYSCALE)
    output_img_cv = cv2.imread("fds.jpg", cv2.IMREAD_GRAYSCALE)
    tmask_cv = cv2.imread("tmask.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Convert to float32 and normalize to [0,1]
    input_image_np = input_image_cv.astype(np.float32) / 255.0
    output_img_np = output_img_cv.astype(np.float32) / 255.0
    tmask_np = tmask_cv.astype(np.float32) / 255.0  # assume tmask is binary: 0 or 1
    
    # Add channel dimension: (H, W) -> (1, H, W)
    input_image_np = np.expand_dims(input_image_np, axis=0)
    output_img_np = np.expand_dims(output_img_np, axis=0)
    tmask_np = np.expand_dims(tmask_np, axis=0)
    
    # Create a batch of size B by repeating the single image
    B = 2
    input_image_np = np.repeat(np.expand_dims(input_image_np, axis=0), B, axis=0)  # shape [B, 1, H, W]
    output_img_np = np.repeat(np.expand_dims(output_img_np, axis=0), B, axis=0)
    tmask_np = np.repeat(np.expand_dims(tmask_np, axis=0), B, axis=0)
    
    # Convert to torch tensors
    input_image_tensor = torch.from_numpy(input_image_np)
    output_img_tensor = torch.from_numpy(output_img_np)
    tmask_tensor = torch.from_numpy(tmask_np)
    
    # Move tensors to device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image_tensor = input_image_tensor.to(device)
    output_img_tensor = output_img_tensor.to(device)
    tmask_tensor = tmask_tensor.to(device)
    
    # Instantiate the LossCalculator (with the provided spline fitter)
    loss_calc = LossCalculator(k=10.0)
    
    # Compute the loss: this passes the network output through the spline fitter,
    # creates a soft mask from the fitted spline, and then computes the overlap loss.
    loss = loss_calc(input_image_tensor, tmask_tensor, output_img_tensor)
    print("Loss:", loss.item())